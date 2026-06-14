import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root, setup_env, run, openvla_dir, resolve_model_path,
    raw_demos_dir, stack_demos_dir,
)

WORKER = "openvla_offline_infer.py"


def _first_episode_dir(root: Path) -> str:
    eps = sorted(p for p in root.glob("episode_*") if p.is_dir())
    if not eps:
        raise SystemExit(f"[075] episode_* not found: {root}")
    return str(eps[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["predict", "eval"], default="predict")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--default-unnorm-key", type=str, default="raccoon_pick_place")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--episode-dir", type=str, default=None)
    p.add_argument("--stack", action="store_true")
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--do-sample", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    model_path = resolve_model_path(rb, args.model_path)

    cmd = [
        sys.executable, WORKER,
        "--mode", args.mode,
        "--model_path", model_path,
        "--default-unnorm-key", args.default_unnorm_key,
        "--device", args.device,
        "--num-steps", str(args.num_steps),
    ]
    if args.do_sample:
        cmd.append("--do-sample")
    if args.instruction:
        cmd += ["--instruction", args.instruction]

    if args.image:
        cmd += ["--image", args.image]
    else:
        if args.episode_dir:
            episode_dir = args.episode_dir
        else:
            root = stack_demos_dir(rb) if args.stack else raw_demos_dir(rb)
            episode_dir = _first_episode_dir(root)
        cmd += ["--episode-dir", episode_dir]

    run(cmd, cwd=ov, extra_env=env)


if __name__ == "__main__":
    main()
