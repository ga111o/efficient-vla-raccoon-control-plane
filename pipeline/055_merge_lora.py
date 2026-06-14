import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, run, openvla_dir  # noqa: E402

WORKER = "vla-scripts/merge_lora.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root_dir", type=str, default=None)
    p.add_argument("--vla_path", type=str, default=None)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)

    # merge_lora.py 가 prismatic 패키지를 import 할 수 있도록 PYTHONPATH 지정.
    env["PYTHONPATH"] = str(ov)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_root_dir = args.run_root_dir or str(ov / "openvla-runs")

    cmd = [sys.executable, WORKER, "--run_root_dir", run_root_dir]
    for flag, val in (("--vla_path", args.vla_path),
                      ("--adapter_dir", args.adapter_dir),
                      ("--run_dir", args.run_dir)):
        if val:
            cmd += [flag, val]

    run(cmd, cwd=ov, extra_env=env)


if __name__ == "__main__":
    main()
