import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root,
    setup_env,
    run,
    raccoon_dataset_dir,
    raw_demos_dir,
    stack_demos_dir,
)

WORKER = "convert_raw_to_openvla_rlds_intermediate.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--joint_pad_dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--include_failed", action="store_true")
    p.add_argument("--drop_idle_steps", action="store_true")
    p.add_argument("--min_joint_delta_norm", type=float, default=None)
    p.add_argument("--min_gripper_delta", type=float, default=None)
    p.add_argument("--min_ee_delta_norm", type=float, default=None)
    p.add_argument("--no_debug_fields", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--stack_raw_root", type=str, default=None)
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)

    def _resolve_path(p_str, default_p):
        if not p_str:
            return default_p
        p = Path(p_str)
        if p.is_absolute():
            return p
        if p.exists():
            return p.resolve()
        worker_rel = (raccoon_dataset_dir(rb) / p).resolve()
        if worker_rel.exists():
            return worker_rel
        return p.resolve()

    raw_root = _resolve_path(args.raw_root, raw_demos_dir(rb))
    out_root = _resolve_path(args.out_root, raccoon_dataset_dir(rb) / "openvla_rlds_intermediate")
    worker = raccoon_dataset_dir(rb) / WORKER

    def _build_cmd(rr, append=False):
        c = [
            sys.executable, str(worker),
            "--raw_root", str(rr),
            "--out_root", str(out_root),
            "--val_ratio", str(args.val_ratio),
        ]
        if args.joint_pad_dim is not None:
            c += ["--joint_pad_dim", str(args.joint_pad_dim)]
        if args.seed is not None:
            c += ["--seed", str(args.seed)]
        if args.include_failed:
            c += ["--include_failed"]
        if args.drop_idle_steps:
            c += ["--drop_idle_steps"]
        if args.min_joint_delta_norm is not None:
            c += ["--min_joint_delta_norm", str(args.min_joint_delta_norm)]
        if args.min_gripper_delta is not None:
            c += ["--min_gripper_delta", str(args.min_gripper_delta)]
        if args.min_ee_delta_norm is not None:
            c += ["--min_ee_delta_norm", str(args.min_ee_delta_norm)]
        if args.no_debug_fields:
            c += ["--no_debug_fields"]
        if append:
            c += ["--append"]
        return c

    tasks = []
    if raw_root.is_dir():
        tasks.append((raw_root, "grasp"))
    else:
        print(f"Warning: Grasp raw root not found: {raw_root}. Skipping grasp task.")

    if args.with_stack or args.stack_raw_root is not None:
        stack_raw = _resolve_path(args.stack_raw_root, stack_demos_dir(rb))
        if stack_raw.is_dir():
            tasks.append((stack_raw, "stack"))
        else:
            print(f"Warning: Stack raw root not found: {stack_raw}. Skipping stack task.")

    if not tasks:
        print(f"Error: No valid raw directories found. Checked:\n  - {raw_root}")
        if args.with_stack or args.stack_raw_root is not None:
            print(f"  - {stack_raw}")
        raise SystemExit(1)

    for i, (rr, name) in enumerate(tasks):
        is_first = (i == 0)
        print(f"\n[020] Processing {name} task (append={not is_first})...")
        run(_build_cmd(rr, append=not is_first), cwd=raccoon_dataset_dir(rb), extra_env=env)


if __name__ == "__main__":
    main()
