import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root, setup_env, run, openvla_dir, resolve_model_path,
)

WORKER = "openvla_server.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--default-unnorm-key", type=str,
                   default="raccoon_pick_place")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--vision-precision", dest="vision_precision", type=str, default="int8")
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    model_path = resolve_model_path(rb, args.model_path)

    cmd = [
        sys.executable, WORKER,
        "--model_path", model_path,
        "--default-unnorm-key", args.default_unnorm_key,
        "--host", args.host,
        "--port", str(args.port),
        "--device", args.device,
        "--precision", args.precision,
        "--vision-precision", args.vision_precision,
    ]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")

    run(cmd, cwd=ov, extra_env=env)


if __name__ == "__main__":
    main()
