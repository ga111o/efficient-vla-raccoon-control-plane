import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root, setup_env, run, openvla_dir, tfds_data_dir, load_dotenv,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla_path", type=str, default="openvla/openvla-7b")
    p.add_argument("--dataset_name", type=str, default="raccoon_pick_place")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--val-interval-steps", dest="val_interval_steps", type=int, default=500)
    p.add_argument("--val-batches", dest="val_batches", type=int, default=10)
    p.add_argument("--run_id_note", type=str, default="raccoon-eef-v100")
    p.add_argument("--data_root_dir", type=str, default=None)
    p.add_argument("--run_root_dir", type=str, default=None)
    p.add_argument("--adapter_tmp_dir", type=str, default=None)
    p.add_argument("--nproc", "--gpus", dest="nproc", type=int, default=1)
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--clearml", dest="clearml", action="store_true", default=None)
    p.add_argument("--no-clearml", dest="clearml", action="store_false")
    p.add_argument("--clearml-project", type=str, default="openvla-raccoon")
    p.add_argument("--merge-in-training", action="store_true")
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--use-8bit-optimizer", action="store_true")
    p.add_argument("--ddp-find-unused-parameters", dest="ddp_find_unused", action="store_true", default=True)
    p.add_argument("--no-ddp-find-unused-parameters", dest="ddp_find_unused", action="store_false")
    args = p.parse_args()

    load_dotenv()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)

    env["PYTHONPATH"] = str(ov)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.wandb:
        env["WANDB_MODE"] = "disabled"

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

    torchrun = shutil.which("torchrun")
    if torchrun:
        launcher = [torchrun]
    else:
        launcher = [sys.executable, "-m", "torch.distributed.run"]

    cmd = launcher + [
        "--standalone", "--nnodes", "1", "--nproc-per-node", str(args.nproc),
        "vla-scripts/finetune.py",
        "--vla_path", args.vla_path,
        "--data_root_dir", data_root_dir,
        "--dataset_name", args.dataset_name,
        "--run_root_dir", run_root_dir,
        "--adapter_tmp_dir", adapter_tmp_dir,
        "--lora_rank", str(args.lora_rank),
        "--batch_size", str(args.batch_size),
        "--grad_accumulation_steps", str(args.grad_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--val_interval_steps", str(args.val_interval_steps),
        "--val_batches", str(args.val_batches),
        "--run_id_note", args.run_id_note,
        "--merge_during_training", "True" if args.merge_in_training else "False",
        "--attn_implementation", args.attn_implementation,
        "--use_8bit_optimizer", "True" if args.use_8bit_optimizer else "False",
        "--ddp_find_unused_parameters", "True" if args.ddp_find_unused else "False",
    ]

    run(cmd, cwd=ov, extra_env=env)


if __name__ == "__main__":
    main()
