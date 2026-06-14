"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import draccus
import torch
# NOTE: import timm (and thus torchvision native libs) BEFORE tensorflow gets pulled in
#       via `prismatic.vla.datasets` below. With newer torch/TF builds, the reverse load
#       order (tensorflow before torchvision/timm) crashes the interpreter with a segfault.
import timm  # noqa: F401  (import-order guard, see comment above)
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    merge_during_training: bool = True                              # Whether to merge LoRA into the backbone in-loop at
                                                                    #   every checkpoint (slow: reloads 7B + writes ~14GB).
                                                                    #   Set False to only save the adapter and merge
                                                                    #   post-hoc once via vla-scripts/merge_lora.py.

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # GPU Memory-saving Arguments (lossless: no quality/speed degradation)
    attn_implementation: str = "flash_attention_2"                  # LLM attention kernel: "flash_attention_2" (memory
                                                                    #   down + faster, exact attention) | "sdpa" | "eager"
                                                                    #   => "eager" reproduces the original (slowest, most mem)
    use_8bit_optimizer: bool = False                                # Use bitsandbytes AdamW8bit for LoRA params (block-wise
                                                                    #   8-bit optimizer states; lossless, frees optim memory)
    ddp_find_unused_parameters: bool = True                         # DDP reducer flag. all-linear LoRA uses every adapter
                                                                    #   each step => False is safe and slightly cheaper

    # Validation Parameters
    val_interval_steps: int = 500                                   # Run validation every N gradient steps (0 disables)
    val_batches: int = 10                                           # Number of batches averaged per validation pass

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@torch.no_grad()
def run_validation(model, val_iter, action_tokenizer, device_id, num_batches):
    """Average loss / action-accuracy / L1 over `num_batches` held-out batches.

    Runs on the unwrapped model (no DDP sync, no grad). `model` is set back to
    train() by the caller. Returns (val_loss, val_accuracy, val_l1)."""
    losses, accuracies, l1_losses = [], [], []
    for _ in range(num_batches):
        batch = next(val_iter)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = model(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
            )

        # Mirror the train-loop accuracy / L1 computation so val is directly comparable.
        action_logits = output.logits[:, model.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        continuous_actions_pred = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
        )
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

        losses.append(output.loss.item())
        accuracies.append(action_accuracy.item())
        l1_losses.append(action_l1_loss.item())

    return (
        sum(losses) / len(losses),
        sum(accuracies) / len(accuracies),
        sum(l1_losses) / len(l1_losses),
    )


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    
    # Append timestamp to ensure unique experiment IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id += f"--{timestamp}"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(
        vla,
        device_ids=[device_id],
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        gradient_as_bucket_view=True,
    )

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_8bit_optimizer:
        # bitsandbytes block-wise 8-bit AdamW: shrinks optimizer state memory (m, v) for the
        #   LoRA params without quality loss (cf. QLoRA). Falls back to fp32 AdamW if unavailable.
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.learning_rate)
            print("[mem] optimizer: bitsandbytes AdamW8bit (8-bit optimizer states)")
        except Exception as e:  # noqa: BLE001
            print(f"[mem] 8-bit optimizer unavailable ({e}); falling back to fp32 AdamW")
            optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # [mem] Static footprint after model + optimizer setup (weights + LoRA grads/states), pre-activation.
    #   Peak (incl. activations) is logged inside the training loop below. Reset peak so the
    #   first measured peak reflects steady-state training, not one-off load spikes.
    if torch.cuda.is_available():
        print(
            f"[mem] attn_impl={cfg.attn_implementation} | post-setup "
            f"allocated={torch.cuda.memory_allocated(device_id) / 1e9:.2f} GB "
            f"reserved={torch.cuda.memory_reserved(device_id) / 1e9:.2f} GB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(device_id)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

        # Record a pointer to this run so post-hoc merge (vla-scripts/merge_lora.py)
        #   and serving can locate the run/adapter dirs without re-deriving exp_id.
        os.makedirs(cfg.run_root_dir, exist_ok=True)
        with open(Path(cfg.run_root_dir) / "last_run.json", "w") as f:
            json.dump(
                {
                    "exp_id": exp_id,
                    "vla_path": cfg.vla_path,
                    "run_dir": str(run_dir),
                    "adapter_dir": str(adapter_dir),
                    "use_lora": cfg.use_lora,
                    "merge_during_training": cfg.merge_during_training,
                },
                f,
                indent=2,
            )

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Held-out Validation Dataset =>> uses the dataset's `val` split (or `train[95%:]` if none defined).
    #   Only the main process runs validation, so only it builds the pipeline. No image_aug, small
    #   shuffle buffer (eval doesn't need heavy shuffling). The loader is infinite (RLDS loops), so we
    #   keep a single iterator and pull `cfg.val_batches` batches per validation pass.
    val_iter = None
    if distributed_state.is_main_process and cfg.val_interval_steps > 0:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=min(cfg.shuffle_buffer_size, 10_000),
            image_aug=False,
            train=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )
        val_iter = iter(val_dataloader)

    # Initialize Logging =>> W&B (+ optional ClearML)
    #   ClearML 은 RB_CLEARML=1 이고 CLEARML_API_* 자격증명이 env 에 있을 때만 켜진다
    #   (050_finetune.py 가 .env 를 읽어 주입). 기본 W&B 는 보통 disabled 라
    #   스칼라를 ClearML 로 직접 report 한다(아래 학습 루프).
    clearml_logger = None
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

        if os.environ.get("RB_CLEARML") == "1":
            try:
                from clearml import Task

                # 매 실행마다 새로운 task 를 만든다: exp_id 는 하이퍼파라미터로 결정되어
                #   재실행 시 동일하므로, task_name 에 실행 타임스탬프를 붙이고
                #   reuse_last_task_id=False 로 직전 task 재사용(기본 동작)을 끈다.
                run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                clearml_task = Task.init(
                    project_name=os.environ.get("CLEARML_PROJECT", "openvla-raccoon"),
                    task_name=f"ft+{exp_id}+{run_stamp}",
                    reuse_last_task_id=False,
                    # 스칼라는 아래에서 수동 report → wandb 자동연동 중복을 피하려 끈다.
                    auto_connect_frameworks=False,
                    output_uri=False,
                )
                clearml_task.connect({k: str(v) for k, v in vars(cfg).items()})
                clearml_logger = clearml_task.get_logger()
                print(f"[ClearML] tracking → {clearml_task.get_output_log_web_page()}")
            except Exception as e:  # noqa: BLE001 (추적 실패가 학습을 막으면 안 됨)
                print(f"[ClearML] disabled (init failed: {e})")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    # [mem/speed] wall-clock time between optimizer steps (no cuda.synchronize → zero throughput
    #   perturbation; async timings average out over the window) for measuring speed regressions.
    recent_step_times = deque(maxlen=20)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        last_step_time = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                # [mem/speed] peak GPU memory (incl. activations) + mean step time, so A/B runs
                #   (eager vs flash_attention_2, fp32 vs 8-bit optim) can be compared quantitatively.
                peak_alloc_gb = torch.cuda.max_memory_allocated(device_id) / 1e9
                peak_reserved_gb = torch.cuda.max_memory_reserved(device_id) / 1e9
                mean_step_s = sum(recent_step_times) / len(recent_step_times) if recent_step_times else 0.0
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        "gpu_peak_alloc_gb": peak_alloc_gb,
                        "gpu_peak_reserved_gb": peak_reserved_gb,
                        "step_time_s": mean_step_s,
                    },
                    step=gradient_step_idx,
                )
                print(
                    f"[mem/speed] step {gradient_step_idx}: peak_alloc={peak_alloc_gb:.2f} GB "
                    f"peak_reserved={peak_reserved_gb:.2f} GB step_time={mean_step_s * 1e3:.0f} ms",
                    flush=True,
                )
                if clearml_logger is not None:
                    clearml_logger.report_scalar("loss", "train_loss", smoothened_loss, gradient_step_idx)
                    clearml_logger.report_scalar(
                        "accuracy", "action_accuracy", smoothened_action_accuracy, gradient_step_idx
                    )
                    clearml_logger.report_scalar("loss", "l1_loss", smoothened_l1_loss, gradient_step_idx)
                    clearml_logger.report_scalar("gpu_mem", "peak_alloc_gb", peak_alloc_gb, gradient_step_idx)
                    clearml_logger.report_scalar("gpu_mem", "peak_reserved_gb", peak_reserved_gb, gradient_step_idx)
                    clearml_logger.report_scalar("speed", "step_time_s", mean_step_s, gradient_step_idx)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                now = time.perf_counter()
                recent_step_times.append(now - last_step_time)
                last_step_time = now
                progress.update()

                # Validation =>> run on held-out split at the optimizer-step boundary (so it fires
                #   once per gradient step, not once per micro-batch under grad accumulation).
                if (
                    val_iter is not None
                    and gradient_step_idx > 0
                    and gradient_step_idx % cfg.val_interval_steps == 0
                ):
                    val_loss, val_accuracy, val_l1_loss = run_validation(
                        vla.module, val_iter, action_tokenizer, device_id, cfg.val_batches
                    )
                    vla.train()  # run_validation does not flip the model back
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_action_accuracy": val_accuracy,
                            "val_l1_loss": val_l1_loss,
                        },
                        step=gradient_step_idx,
                    )
                    print(
                        f"[val] step {gradient_step_idx}: loss={val_loss:.4f} "
                        f"acc={val_accuracy:.4f} l1={val_l1_loss:.4f}",
                        flush=True,
                    )
                    if clearml_logger is not None:
                        # Same graph titles as train ("loss"/"accuracy") → val plots on the same axes.
                        clearml_logger.report_scalar("loss", "val_loss", val_loss, gradient_step_idx)
                        clearml_logger.report_scalar("accuracy", "val_action_accuracy", val_accuracy, gradient_step_idx)
                        clearml_logger.report_scalar("loss", "val_l1_loss", val_l1_loss, gradient_step_idx)

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                #   =>> When `merge_during_training` is False we skip this entirely and only keep the
                #       adapter; run vla-scripts/merge_lora.py once after training instead.
                if cfg.use_lora and cfg.merge_during_training:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    if cfg.use_lora and not cfg.merge_during_training and distributed_state.is_main_process:
        print(
            "\n[finetune] Training done. LoRA adapter saved (merge was deferred).\n"
            f"[finetune]   adapter : {adapter_dir}\n"
            f"[finetune]   run_dir : {run_dir}\n"
            "[finetune] Run the post-hoc merge once to produce a servable full model:\n"
            f"[finetune]   python vla-scripts/merge_lora.py --run_root_dir {cfg.run_root_dir}\n",
            flush=True,
        )


if __name__ == "__main__":
    finetune()
