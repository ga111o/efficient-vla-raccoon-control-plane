import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, run, mujoco_dir  # noqa: E402

WORKER = "raccoon_grasp_multicolor_scene_dataset.py"
DEFAULT_DATASET_ROOT = "raccoon_pick_and_stack"


def _resolve_num_workers(spec: str, num_episodes: int | None) -> int:
    if spec.lower() == "auto":
        cores = os.cpu_count() or 1
        cap = num_episodes if num_episodes is not None else 400
        return max(1, min(cores, cap))
    try:
        n = int(spec)
    except ValueError:
        raise SystemExit(f"[015] --num-workers Value Err: {spec!r}")
    if n < 1:
        raise SystemExit("[015] --num-workers > 0")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--xml-path", type=str, default=None)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--base-label", type=str, default=None)
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--settle-seconds", type=float, default=None)
    p.add_argument("--noslip-iterations", type=int, default=None)
    p.add_argument("--camera-jitter", type=float, default=0.02)
    p.add_argument("--no-clean", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    cwd = mujoco_dir(rb)

    num_workers = _resolve_num_workers(args.num_workers, args.num_episodes)
    if args.num_workers.lower() == "auto":
        print(f"[015] --num-workers auto -> {num_workers} "
              f"(cpu_count={os.cpu_count()}, num_episodes={args.num_episodes or 400})")

    ds_root_name = args.dataset_root or DEFAULT_DATASET_ROOT
    ds_root = (cwd / ds_root_name) if not Path(ds_root_name).is_absolute() else Path(ds_root_name)

    if not args.no_clean and ds_root.is_dir():
        removed = 0
        t0 = time.monotonic()
        for ep in ds_root.glob("episode*"):
            if ep.is_dir():
                shutil.rmtree(ep)
            else:
                ep.unlink()
            removed += 1
        print(f"[clean] removed {removed} existing episode* under {ds_root} "
              f"(+ {time.monotonic() - t0:.1f}s)")

    cmd = [sys.executable, str(cwd / WORKER), "--task", "stack",
           "--num-workers", str(num_workers)]
    if args.num_episodes is not None:
        cmd += ["--num-episodes", str(args.num_episodes)]
    if args.xml_path is not None:
        cmd += ["--xml-path", args.xml_path]
    if args.dataset_root is not None:
        cmd += ["--dataset-root", args.dataset_root]
    if args.base_label is not None:
        cmd += ["--base-label", args.base_label]
    if args.keep_failed:
        cmd += ["--keep-failed"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.speed is not None:
        cmd += ["--speed", str(args.speed)]
    if args.hz is not None:
        cmd += ["--hz", str(args.hz)]
    if args.settle_seconds is not None:
        cmd += ["--settle-seconds", str(args.settle_seconds)]
    if args.noslip_iterations is not None:
        cmd += ["--noslip-iterations", str(args.noslip_iterations)]
    cmd += ["--camera-jitter", str(args.camera_jitter)]

    run(cmd, cwd=cwd, extra_env=env)


if __name__ == "__main__":
    main()
