import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    resolve_rb_root, setup_env, run, openvla_dir, mujoco_dir, resolve_model_path,
)

SERVER = "openvla_server.py"
WORKER = "openvla_closed_loop_eval.py"


def _wait_health(server_url, proc, timeout):
    health = f"{server_url}/health"
    deadline = time.time() + timeout
    print(f"[080] server health wait: {health} (timeout={timeout}s)", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"policy server exited during startup (exit={proc.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    print("[080] server ready.", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2.0)
    raise TimeoutError(f"policy server health timeout ({timeout}s): {health}")


def _terminate(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--default-unnorm-key", type=str, default="raccoon_pick_place")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-startup-timeout", type=int, default=600)
    p.add_argument("--xml-path", type=str, default=None)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--hz", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-path", type=str, default=None)
    p.add_argument("--render-video", type=str, default=None)
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)
    mj = mujoco_dir(rb)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ov), str(mj), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    model_path = resolve_model_path(rb, args.model_path)
    xml_path = args.xml_path or str(mj / "Raccoon_colored_cylinder.xml")
    report_path = args.report_path or str(
        Path(__file__).resolve().parent / "outputs" / "eval" / "eval_report.json")
    server_url = f"http://{args.host}:{args.port}"

    server_cmd = [
        sys.executable, SERVER,
        "--model_path", model_path,
        "--default-unnorm-key", args.default_unnorm_key,
        "--device", args.device,
        "--host", args.host,
        "--port", str(args.port),
    ]
    print(f"\n[080] starting policy server (cwd={ov})\n      $ {' '.join(server_cmd)}\n",
          flush=True)
    server = subprocess.Popen(server_cmd, cwd=str(ov), env=env)

    try:
        _wait_health(server_url, server, args.server_startup_timeout)

        worker_cmd = [
            sys.executable, WORKER,
            "--server-url", server_url,
            "--model_path", model_path,
            "--default-unnorm-key", args.default_unnorm_key,
            "--xml-path", xml_path,
            "--num-episodes", str(args.num_episodes),
            "--max-steps", str(args.max_steps),
            "--hz", str(args.hz),
            "--seed", str(args.seed),
            "--report-path", report_path,
        ]
        if args.render_video:
            worker_cmd += ["--render-video", args.render_video]

        run(worker_cmd, cwd=ov, extra_env=env)
    finally:
        print("[080] stopping policy server.", flush=True)
        _terminate(server)


if __name__ == "__main__":
    main()
