import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root, setup_env, run, openvla_dir, tfds_data_dir, load_dotenv,
)

WORKER = "vla-scripts/finetune_sandwich.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla_path", type=str, default="openvla/openvla-7b")
    p.add_argument("--dataset_name", type=str, default="raccoon_pick_place")
    p.add_argument("--quant-backend", dest="quant_backend", type=str, default="torchao",
                   choices=["torchao", "bitsandbytes"])
    p.add_argument("--backbone-precision", dest="backbone_precision", type=str, default="fp8")
    p.add_argument("--vision-precision", dest="vision_precision", type=str, default="bf16")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--vision_lora_rank", type=int, default=16)
    p.add_argument("--llm-lora-mode", dest="llm_lora_mode", type=str, default="all",
                   choices=["all", "qv"])
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--use_rslora", dest="use_rslora", action="store_true", default=False)
    p.add_argument("--no-vision-lora", dest="train_vision_lora", action="store_false", default=True)
    p.add_argument("--no-train-projector", dest="train_projector", action="store_false", default=True)
    p.add_argument("--no-train-action-head", dest="train_action_head", action="store_false", default=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation_steps", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=11000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--val_interval_steps", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=10)
    p.add_argument("--attn-implementation", dest="attn_implementation", type=str,
                   default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false", default=True)
    p.add_argument("--no-8bit-optimizer", dest="use_8bit_optimizer",
                   action="store_false", default=True)
    p.add_argument("--use-fast-tokenizer", dest="use_fast_tokenizer", action="store_true")
    p.add_argument("--data_root_dir", type=str, default=None)
    p.add_argument("--run_root_dir", type=str, default=None)
    p.add_argument("--adapter_tmp_dir", type=str, default=None)
    p.add_argument("--run_id_note", type=str, default="sandwich-fp8")
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument("--clearml", dest="clearml", action="store_true", default=None)
    p.add_argument("--no-clearml", dest="clearml", action="store_false")
    p.add_argument("--clearml-project", type=str, default="openvla-raccoon")
    args = p.parse_args()

    load_dotenv()
    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)
    env["PYTHONPATH"] = str(ov)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("WANDB_MODE", "disabled")

    creds_present = bool(
        os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get("CLEARML_API_SECRET_KEY")
    )
    use_clearml = creds_present if args.clearml is None else args.clearml
    if use_clearml:
        if not creds_present:
            print("[ClearML] warning: CLEARML credentials missing in .env")
        env["RB_CLEARML"] = "1"
        env["CLEARML_PROJECT"] = args.clearml_project
        print(f"[ClearML] enabled project={args.clearml_project}")
    else:
        env["RB_CLEARML"] = "0"
        print("[ClearML] disabled")

    data_root_dir = args.data_root_dir or str(tfds_data_dir(rb))
    run_root_dir = args.run_root_dir or str(ov / "openvla-runs")
    adapter_tmp_dir = args.adapter_tmp_dir or str(ov / "openvla-adapter-tmp")

    def b(v):
        return "True" if v else "False"

    cmd = [
        sys.executable, WORKER,
        "--vla_path", args.vla_path,
        "--data_root_dir", data_root_dir,
        "--dataset_name", args.dataset_name,
        "--run_root_dir", run_root_dir,
        "--adapter_tmp_dir", adapter_tmp_dir,
        "--quant_backend", args.quant_backend,
        "--backbone_precision", args.backbone_precision,
        "--vision_precision", args.vision_precision,
        "--lora_rank", str(args.lora_rank),
        "--vision_lora_rank", str(args.vision_lora_rank),
        "--llm_lora_mode", args.llm_lora_mode,
        "--use_rslora", b(args.use_rslora),
        "--train_vision_lora", b(args.train_vision_lora),
        "--train_projector", b(args.train_projector),
        "--train_action_head", b(args.train_action_head),
        "--batch_size", str(args.batch_size),
        "--grad_accumulation_steps", str(args.grad_accumulation_steps),
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--learning_rate", str(args.learning_rate),
        "--val_interval_steps", str(args.val_interval_steps),
        "--val_batches", str(args.val_batches),
        "--attn_implementation", args.attn_implementation,
        "--gradient_checkpointing", b(args.gradient_checkpointing),
        "--use_8bit_optimizer", b(args.use_8bit_optimizer),
        "--use_fast_tokenizer", b(args.use_fast_tokenizer),
        "--run_id_note", args.run_id_note,
    ]
    if args.lora_alpha is not None:
        cmd += ["--lora_alpha", str(args.lora_alpha)]

    run(cmd, cwd=ov, extra_env=env)


if __name__ == "__main__":
    main()
