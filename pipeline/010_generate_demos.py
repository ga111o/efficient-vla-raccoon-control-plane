import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, run, mujoco_dir  # noqa: E402

WORKER = "raccoon_grasp_multicolor_scene_dataset.py"


def _resolve_num_workers(spec: str, num_episodes: int | None) -> int:
    if spec.lower() == "auto":
        cores = os.cpu_count() or 1
        cap = num_episodes if num_episodes is not None else 400
        return max(1, min(cores, cap))
    try:
        n = int(spec)
    except ValueError:
        raise SystemExit(f"[010] --num-workers Value Err: {spec!r}")
    if n < 1:
        raise SystemExit("[010] --num-workers > 0")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--xml-path", type=str, default=None)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-clean", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    cwd = mujoco_dir(rb)

    num_workers = _resolve_num_workers(args.num_workers, args.num_episodes)
    if args.num_workers.lower() == "auto":
        print(f"[010] --num-workers auto -> {num_workers} "
              f"(cpu_count={os.cpu_count()}, num_episodes={args.num_episodes or 400})")

    ds_root_name = args.dataset_root or "raccoon_grasp_colored_cylinder"
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

    cmd = [sys.executable, str(cwd / WORKER), "--num-workers", str(num_workers)]
    if args.num_episodes is not None:
        cmd += ["--num-episodes", str(args.num_episodes)]
    if args.xml_path is not None:
        cmd += ["--xml-path", args.xml_path]
    if args.dataset_root is not None:
        cmd += ["--dataset-root", args.dataset_root]
    if args.keep_failed:
        cmd += ["--keep-failed"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    run(cmd, cwd=cwd, extra_env=env)


if __name__ == "__main__":
    main()
