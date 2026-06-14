import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (
    resolve_rb_root,
    setup_env,
    mujoco_dir,
    raccoon_dataset_dir,
    rlds_builder_dir,
    openvla_dir,
    tfds_data_dir,
)


def _check_import(label, code):
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
        )
        print(f"  [ok ] {label}: {out.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {label}: {e.stderr.strip().splitlines()[-1] if e.stderr.strip() else e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    setup_env(rb)

    print("\n[paths]")
    print("  MUJOCO_DIR          =", mujoco_dir(rb))
    print("  RACCOON_DATASET_DIR =", raccoon_dataset_dir(rb))
    print("  RLDS_BUILDER_DIR    =", rlds_builder_dir(rb))
    print("  OPENVLA_DIR         =", openvla_dir(rb))
    print("  TFDS_DATA_DIR       =", tfds_data_dir(rb))

    print("\n[imports] (python =", sys.executable, ")")
    _check_import("torch", "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())")
    _check_import("tensorflow", "import tensorflow as tf; print(tf.__version__)")
    _check_import("dlimp", "import dlimp; print('dlimp ok')")
    _check_import("apache-beam", "import apache_beam; print('beam ok')")
    _check_import("tomli", "import tomli; print('tomli ok')")

    if not args.skip_gpu:
        smi = shutil.which("nvidia-smi")
        if smi:
            print("\n[nvidia-smi]")
            subprocess.run([smi], check=False)
        else:
            print("\nnvidia-smi not found")


if __name__ == "__main__":
    main()
