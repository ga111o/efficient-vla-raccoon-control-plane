import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, run, openvla_dir  # noqa: E402

DEFAULT_REPO = "fair-lab/openvla-7b-finetuned-raccoonbot"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=DEFAULT_REPO)
    p.add_argument("--local-dir", type=str, default=None)
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)

    local_dir = args.local_dir or str(openvla_dir(rb) / "openvla-runs" / args.repo.split("/")[-1])

    hf_bin = shutil.which("hf")
    if hf_bin:
        cmd = [hf_bin, "download", args.repo, "--local-dir", local_dir]
    else:
        cmd = [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
               "download", args.repo, "--local-dir", local_dir]

    run(cmd, extra_env=env)
    print(f"[downloaded] {args.repo} -> {local_dir}")


if __name__ == "__main__":
    main()
