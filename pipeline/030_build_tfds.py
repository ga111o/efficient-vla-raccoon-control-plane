import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, run, rlds_builder_dir, tfds_data_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--data-dir", type=str, default=None)
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    cwd = rlds_builder_dir(rb) / "raccoon_pick_place"

    data_dir = args.data_dir or str(tfds_data_dir(rb))
    env["TFDS_DATA_DIR"] = data_dir

    tfds_bin = shutil.which("tfds")
    if tfds_bin:
        cmd = [tfds_bin, "build"]
    else:
        cmd = [sys.executable, "-m", "tensorflow_datasets.scripts.cli.main", "build"]

    if not args.no_overwrite:
        cmd += ["--overwrite"]
    cmd += ["--data_dir", data_dir]

    run(cmd, cwd=cwd, extra_env=env)


if __name__ == "__main__":
    main()
