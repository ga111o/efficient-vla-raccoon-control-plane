"""
finetune_sandwich.py — OpenVLA "샌드위치(Sandwich)" QLoRA 파인튜닝 (단일 RTX 5080 16GB).

기존 vla-scripts/finetune.py 는 `target_modules="all-linear"` 로 LLM·projector·vision 에
모두 LoRA 를 붙인다(=projector 까지 건드림). 이 스크립트는 과제 §2~§3 의 **비대칭 혼합
정밀도 + 샌드위치 학습**을 구현한다:

  ┌─ Vision encoder : BF16 (옵션 LoRA r=16) ────────── 동결 해제, 가볍게 미세조정
  │  Projector      : BF16 전체 가중치 (modules_to_save) ─ 절대 양자화 금지, full FT
  │  LLM backbone   : FP8 동결(freeze) + LoRA r=32 (BF16 어댑터) ─ VRAM 절감 핵심
  └─ Action head    : lm_head BF16 전체 가중치 (modules_to_save) ─ 절대 양자화 금지, full FT
                       (= "샌드위치": 양자화된 LLM 을 BF16 모듈들이 위아래로 감싼다)

정밀도/양자화 정책은 prismatic.quant.policy 가 단독 책임진다(서빙과 공유). 이 스크립트는
데이터 로딩·학습 루프만 담당한다. 단일 GPU 전제(DDP 미사용 — torchao FP8 와의 호환을
단순화). 멀티 GPU 가 필요하면 기존 050_finetune.py(all-linear, DDP)를 쓴다.

실행:
    python vla-scripts/finetune_sandwich.py \
        --vla_path openvla/openvla-7b \
        --data_root_dir <TFDS_ROOT> --dataset_name raccoon_pick_place \
        --run_root_dir <RUNS> --adapter_tmp_dir <ADAPTERS> \
        --backbone_precision fp8 --quant_backend torchao \
        --lora_rank 32 --vision_lora_rank 16 \
        --batch_size 4 --grad_accumulation_steps 4 --max_steps 11000 --save_steps 1000

학습이 끝나면 어댑터(+modules_to_save 의 projector/lm_head)가 adapter_dir 에 저장된다.
서빙용 단일 체크포인트가 필요하면 vla-scripts/merge_lora.py 로 BF16 base 에 병합한다
(병합은 FP8 가 아닌 원본 BF16 base 에서 하므로 무손실).
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
# timm 을 tensorflow(아래 prismatic.vla.datasets)보다 먼저 import (finetune.py 와 동일한 순서 가드).
import timm  # noqa: F401
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from prismatic.quant.policy import (
    QuantPolicy,
    apply_torchao_quantization,
    build_bnb_config,
    build_lora_config,
    summarize_precision,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class SandwichConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # 경로
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "raccoon_pick_place"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # 양자화 정책 (prismatic.quant.policy)
    quant_backend: str = "torchao"                # "torchao" | "bitsandbytes"
    backbone_precision: str = "fp8"               # torchao: fp8(weight-only, 동결) ; bnb: int8|nf4
    vision_precision: str = "bf16"                # 학습 중엔 BF16 (서빙 시 INT8 PTQ). int8/int4 가능

    # 샌드위치 학습 구성 (§3)
    lora_rank: int = 32                           # LLM Q,V+선형 LoRA rank
    vision_lora_rank: int = 16                    # vision LoRA rank (train_vision_lora 시)
    lora_dropout: float = 0.0
    llm_lora_mode: str = "all"                    # "all"(검증 레시피) | "qv"(스펙 최소)
    lora_alpha: Optional[int] = None              # LLM LoRA alpha (None=현행 min(rank,16)). capacity 레버: α↑ → γ=α/r↑ (VRAM 증가 0)
    use_rslora: bool = False                      # rsLoRA 스케일(α/√r) — under-fit 시 적응 강도↑ (VRAM 증가 0)
    train_vision_lora: bool = True                # vision 에 LoRA r=16 부착 (False 면 vision 완전 동결)
    train_projector: bool = True                  # projector full BF16 학습 (필수 — 기본 True)
    train_action_head: bool = True                # lm_head full BF16 학습 (필수 — 기본 True)

    # 학습 하이퍼파라미터
    batch_size: int = 4                           # 16GB 기준 보수적 기본값
    grad_accumulation_steps: int = 4
    max_steps: int = 11_000
    save_steps: int = 1_000
    learning_rate: float = 5e-4                    # cosine 스케줄의 peak LR (warmup 후 도달)
    warmup_steps: int = 100                        # cosine LR warmup 스텝 (≈max_steps의 2~3%)
    max_grad_norm: float = 1.0                     # gradient clipping 임계값 (발산 스파이크 차단)
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    # 메모리 절감
    attn_implementation: str = "flash_attention_2"   # "flash_attention_2"|"sdpa"|"eager"
    gradient_checkpointing: bool = True              # 활성 메모리↓ (속도 약간↓). 16GB 권장.
    use_8bit_optimizer: bool = True                  # projector/lm_head full FT optim state↓ (bnb AdamW8bit)

    # 검증
    val_interval_steps: int = 500
    val_batches: int = 10

    # 토크나이저 최적화 (§3, 권장)
    use_fast_tokenizer: bool = False                 # FAST action tokenizer (현재 문서화된 확장점 — 아래 경고 참고)

    # 추적
    run_id_note: Optional[str] = "sandwich-fp8"
    # fmt: on


@torch.no_grad()
def run_validation(model, val_iter, action_tokenizer, num_patches, device, num_batches):
    """held-out 배치들에 대한 (loss, action accuracy, L1) 평균. 학습 루프와 동일 계산."""
    losses, accuracies, l1_losses = [], [], []
    for _ in range(num_batches):
        batch = next(val_iter)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"],
            )
        action_logits = output.logits[:, num_patches:-1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()
        cont_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
        cont_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
        l1 = torch.nn.functional.l1_loss(cont_pred, cont_gt)
        losses.append(output.loss.item())
        accuracies.append(action_accuracy.item())
        l1_losses.append(l1.item())
    n = len(losses)
    return sum(losses) / n, sum(accuracies) / n, sum(l1_losses) / n


@draccus.wrap()
def finetune(cfg: SandwichConfig) -> None:
    print(f"[sandwich] Fine-tuning `{cfg.vla_path}` on `{cfg.dataset_name}` "
          f"(backend={cfg.quant_backend}, backbone={cfg.backbone_precision}, vision={cfg.vision_precision})")
    assert torch.cuda.is_available(), "샌드위치 파인튜닝은 GPU(RTX 5080) 가 필요합니다."
    device = "cuda"
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if cfg.use_fast_tokenizer:
        print("[sandwich][WARN] --use_fast_tokenizer 는 256-bin action head 경로와 분리된 "
              "문서화된 확장점입니다(QUANTIZATION.md 참고). 이번 실행은 기본 256-bin "
              "ActionTokenizer 로 진행합니다(OFT 메모리 절감 옵션은 그대로 적용).")

    # 실험 ID
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+sandwich+{cfg.quant_backend}-{cfg.backbone_precision}"
        f"+lora-r{cfg.lora_rank}+vis-r{cfg.vision_lora_rank if cfg.train_vision_lora else 'off'}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
    )
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    exp_id += f"--{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # HF Auto 클래스에 OpenVLA 등록
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    policy = QuantPolicy(
        backend=cfg.quant_backend,
        backbone_precision=cfg.backbone_precision,
        vision_precision=cfg.vision_precision,
    ).validate()

    # === 모델 로드 + 비대칭 양자화 ===
    if cfg.quant_backend == "torchao":
        # BF16 전체를 16GB GPU 에 먼저 올리면 OOM(≈14.5GB) → CPU 로 로드 후
        # quantize_(device="cuda") 가 layer-by-layer 로 FP8 변환하며 GPU 로 올린다.
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        report = apply_torchao_quantization(vla, policy, device=device)
        vla = vla.to(device)  # 양자화 안 된 나머지(vision/projector/lm_head/embed)를 GPU 로
    elif cfg.quant_backend == "bitsandbytes":
        bnb_cfg = build_bnb_config(policy)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16, quantization_config=bnb_cfg,
            low_cpu_mem_usage=True, trust_remote_code=True, device_map={"": device},
        )
        report = {"backend": "bitsandbytes", "backbone": cfg.backbone_precision,
                  "vision/projector/lm_head": "bf16 (skip_modules)"}
        from peft import prepare_model_for_kbit_training
        vla = prepare_model_for_kbit_training(
            vla, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    else:
        raise ValueError(f"quant_backend={cfg.quant_backend!r} (torchao|bitsandbytes)")

    print("[sandwich] 양자화 적용 리포트:", json.dumps(report, ensure_ascii=False))

    # 비대칭 정밀도가 실제로 먹었는지 + 양자화 후 실제 weight 메모리 확인 (PEFT 래핑 전 =
    # base_model. 접두사/modules_to_save 중복이 없는 깔끔한 그룹 분해).
    print(summarize_precision(vla))

    # 액션 logits 슬라이스 오프셋(=vision patch 수)은 PEFT 래핑 전에 int 로 확보.
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    image_sizes = tuple(vla.config.image_sizes)

    # === 샌드위치 LoRA 구성 (PEFT) ===
    lora_config = build_lora_config(
        vla,
        llm_rank=cfg.lora_rank,
        vision_rank=cfg.vision_lora_rank,
        lora_dropout=cfg.lora_dropout,
        llm_mode=cfg.llm_lora_mode,
        lora_alpha=cfg.lora_alpha,
        use_rslora=cfg.use_rslora,
        train_vision_lora=cfg.train_vision_lora,
        train_projector=cfg.train_projector,
        train_action_head=cfg.train_action_head,
    )
    from peft import get_peft_model
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        # frozen embedding 을 통과하는 grad 가 흐르도록(checkpointing 필수 전제).
        vla.enable_input_require_grads()
        try:
            vla.gradient_checkpointing_enable()
        except Exception as e:  # noqa: BLE001
            print(f"[sandwich][WARN] gradient_checkpointing_enable 실패: {e}")

    vla.train()

    print(f"[mem] post-setup allocated={torch.cuda.memory_allocated(device)/1e9:.2f} GB "
          f"reserved={torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats(device)

    # === Optimizer (trainable: LoRA + projector full + lm_head full) ===
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if cfg.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.learning_rate)
            print("[mem] optimizer: bitsandbytes AdamW8bit (8-bit optimizer states)")
        except Exception as e:  # noqa: BLE001
            print(f"[mem] 8-bit optimizer unavailable ({e}); fp32 AdamW 로 폴백")
            optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # === LR 스케줄러 (cosine + warmup) ===
    # 상수 LR 은 작은 유효배치(16)에서 좋은 basin 진입 후 발산한다 → warmup 으로 부드럽게
    # 진입하고 cosine decay 로 후반 LR 을 낮춰 최적점에 머물게 한다. num_training_steps 는
    # 실제 실행 길이(max_steps)와 일치해야 decay 가 ~0 까지 완주한다(짧게 끊으면 LR 이 덜 내려감).
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )
    print(f"[sched] cosine warmup={cfg.warmup_steps} total={cfg.max_steps} "
          f"peak_lr={cfg.learning_rate:.2e} max_grad_norm={cfg.max_grad_norm}", flush=True)

    # === 데이터셋 ===
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=image_sizes, shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    os.makedirs(cfg.run_root_dir, exist_ok=True)
    with open(Path(cfg.run_root_dir) / "last_run.json", "w") as f:
        json.dump({
            "exp_id": exp_id, "vla_path": cfg.vla_path,
            "run_dir": str(run_dir), "adapter_dir": str(adapter_dir),
            "use_lora": True, "merge_during_training": False,
            "sandwich": True, "quant_backend": cfg.quant_backend,
            "backbone_precision": cfg.backbone_precision,
        }, f, indent=2)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, sampler=None,
                            collate_fn=collator, num_workers=0)

    val_iter = None
    if cfg.val_interval_steps > 0:
        val_dataset = RLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform,
            resize_resolution=image_sizes,
            shuffle_buffer_size=min(cfg.shuffle_buffer_size, 10_000),
            image_aug=False, train=False,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=None,
                                    collate_fn=collator, num_workers=0)
        val_iter = iter(val_dataloader)

    # === 실험 추적 (ClearML) ===
    #   RB_CLEARML=1 이고 CLEARML_API_* 자격증명이 env 에 있을 때만 켜진다
    #   (051_finetune_sandwich.py 가 .env 를 읽어 주입). 단일 GPU 라 wandb 는 쓰지 않고
    #   스칼라를 ClearML 로 직접 report 한다(아래 학습/검증 루프).
    clearml_logger = None
    if os.environ.get("RB_CLEARML") == "1":
        try:
            from clearml import Task

            # exp_id 에 이미 타임스탬프가 붙어 매 실행이 고유하지만, reuse_last_task_id=False
            # 로 직전 task 재사용(기본 동작)을 명시적으로 끈다.
            clearml_task = Task.init(
                project_name=os.environ.get("CLEARML_PROJECT", "openvla-raccoon"),
                task_name=f"sandwich+{exp_id}",
                reuse_last_task_id=False,
                auto_connect_frameworks=False,  # 스칼라는 아래에서 수동 report.
                output_uri=False,
            )
            clearml_task.connect({k: str(v) for k, v in vars(cfg).items()})
            clearml_logger = clearml_task.get_logger()
            print(f"[ClearML] tracking → {clearml_task.get_output_log_web_page()}")
        except Exception as e:  # noqa: BLE001 (추적 실패가 학습을 막으면 안 됨)
            print(f"[ClearML] disabled (init failed: {e})")

    # === 학습 루프 ===
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_acc = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1 = deque(maxlen=cfg.grad_accumulation_steps)
    recent_step_times = deque(maxlen=20)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        optimizer.zero_grad()
        last_step_time = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                )
                loss = output.loss
            (loss / cfg.grad_accumulation_steps).backward()

            action_logits = output.logits[:, num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            action_accuracy = ((action_preds == action_gt) & mask).sum().float() / mask.sum().float()
            cont_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
            cont_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
            action_l1 = torch.nn.functional.l1_loss(cont_pred, cont_gt)

            recent_losses.append(loss.item())
            recent_acc.append(action_accuracy.item())
            recent_l1.append(action_l1.item())
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            if gradient_step_idx % 10 == 0:
                peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
                peak_res = torch.cuda.max_memory_reserved(device) / 1e9
                mean_step = sum(recent_step_times) / len(recent_step_times) if recent_step_times else 0.0
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_acc = sum(recent_acc) / len(recent_acc)
                smoothened_l1 = sum(recent_l1) / len(recent_l1)
                cur_lr = lr_scheduler.get_last_lr()[0]
                print(f"[step {gradient_step_idx}] loss={smoothened_loss:.4f} "
                      f"acc={smoothened_acc:.4f} l1={smoothened_l1:.4f} "
                      f"peak_alloc={peak_alloc:.2f}GB peak_res={peak_res:.2f}GB step={mean_step*1e3:.0f}ms "
                      f"lr={cur_lr:.2e}",
                      flush=True)
                if clearml_logger is not None:
                    clearml_logger.report_scalar("loss", "train_loss", smoothened_loss, gradient_step_idx)
                    clearml_logger.report_scalar("accuracy", "action_accuracy", smoothened_acc, gradient_step_idx)
                    clearml_logger.report_scalar("loss", "l1_loss", smoothened_l1, gradient_step_idx)
                    clearml_logger.report_scalar("gpu_mem", "peak_alloc_gb", peak_alloc, gradient_step_idx)
                    clearml_logger.report_scalar("gpu_mem", "peak_reserved_gb", peak_res, gradient_step_idx)
                    clearml_logger.report_scalar("speed", "step_time_s", mean_step, gradient_step_idx)
                    clearml_logger.report_scalar("lr", "learning_rate", cur_lr, gradient_step_idx)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                now = time.perf_counter()
                recent_step_times.append(now - last_step_time)
                last_step_time = now
                progress.update()

                if (val_iter is not None and gradient_step_idx > 0
                        and gradient_step_idx % cfg.val_interval_steps == 0):
                    vloss, vacc, vl1 = run_validation(
                        vla, val_iter, action_tokenizer, num_patches, device, cfg.val_batches
                    )
                    vla.train()
                    print(f"[val step {gradient_step_idx}] loss={vloss:.4f} acc={vacc:.4f} l1={vl1:.4f}", flush=True)
                    if clearml_logger is not None:
                        # train 과 같은 그래프 제목("loss"/"accuracy") → 같은 축에 겹쳐 그린다.
                        clearml_logger.report_scalar("loss", "val_loss", vloss, gradient_step_idx)
                        clearml_logger.report_scalar("accuracy", "val_action_accuracy", vacc, gradient_step_idx)
                        clearml_logger.report_scalar("loss", "val_l1_loss", vl1, gradient_step_idx)

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0 \
                    and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                print(f"[sandwich] saving adapter (+projector/lm_head) @ step {gradient_step_idx} -> {adapter_dir}")
                processor.save_pretrained(run_dir)
                vla.save_pretrained(adapter_dir)   # LoRA + modules_to_save(projector,lm_head)

            if gradient_step_idx == cfg.max_steps:
                print(f"[sandwich] max step {cfg.max_steps} reached — stopping.")
                break

    # 최종 저장
    processor.save_pretrained(run_dir)
    vla.save_pretrained(adapter_dir)
    print(
        "\n[sandwich] 학습 완료. 어댑터(+projector/lm_head full weights)가 저장됨:\n"
        f"  adapter : {adapter_dir}\n  run_dir : {run_dir}\n"
        "서빙용 단일 체크포인트가 필요하면 BF16 base 에 병합:\n"
        f"  python vla-scripts/merge_lora.py --run_root_dir {cfg.run_root_dir}\n"
        "그 뒤 FP8 W8A8 서빙: openvla_server.py --precision fp8_w8a8 (또는 070_serve.py --precision fp8_w8a8)\n",
        flush=True,
    )


if __name__ == "__main__":
    finetune()
