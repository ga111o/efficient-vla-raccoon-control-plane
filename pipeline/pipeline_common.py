import json
import os
import shlex
import subprocess
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PIPELINE_DIR = _THIS_FILE.parent
_REPO_ROOT = _PIPELINE_DIR.parent


def resolve_rb_root() -> Path:
    env_rb = os.environ.get("RB_ROOT")
    if env_rb:
        return Path(env_rb).resolve()

    candidate = _REPO_ROOT / "Raccoonbot_Openvla"
    if candidate.is_dir():
        return candidate.resolve()

    cwd = Path.cwd()
    if (cwd / "Raccoonbot_Openvla").is_dir():
        return (cwd / "Raccoonbot_Openvla").resolve()
    if cwd.name == "Raccoonbot_Openvla":
        return cwd.resolve()
    if (cwd.parent / "Raccoonbot_Openvla").is_dir():
        return (cwd.parent / "Raccoonbot_Openvla").resolve()

    raise RuntimeError(
        "Raccoonbot_Openvla not found. Set RB_ROOT or run from repo. "
        f"(search: {_REPO_ROOT}, cwd={cwd})"
    )


def setup_env(rb_root: Path) -> dict:
    rb_root = Path(rb_root).resolve()
    os.environ["RB_ROOT"] = str(rb_root)
    os.environ["TFDS_DATA_DIR"] = str(tfds_data_dir(rb_root))

    print("RB_ROOT       =", os.environ["RB_ROOT"])
    print("TFDS_DATA_DIR =", os.environ["TFDS_DATA_DIR"])

    return dict(os.environ)


def load_dotenv(path=None) -> dict:
    env_path = Path(path) if path else (_REPO_ROOT / ".env")
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        loaded[key] = val
        os.environ.setdefault(key, val)
    return loaded


def run(cmd, cwd=None, extra_env=None):
    cmd = [str(c) for c in cmd]
    env = None
    if extra_env is not None:
        env = dict(os.environ)
        env.update(extra_env)

    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[run] (cwd={cwd or os.getcwd()})\n      $ {printable}\n", flush=True)

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
    )


def mujoco_dir(rb_root: Path) -> Path:
    return Path(rb_root) / "Mujoco"


def raccoon_dataset_dir(rb_root: Path) -> Path:
    return mujoco_dir(rb_root) / "raccoon_dataset"


def rlds_builder_dir(rb_root: Path) -> Path:
    return mujoco_dir(rb_root) / "rlds_dataset_builder"


def openvla_dir(rb_root: Path) -> Path:
    return Path(rb_root) / "openvla"


def resolve_model_path(rb_root: Path, explicit: str | None = None) -> str:
    if explicit:
        return str(Path(explicit).resolve())

    run_root = openvla_dir(rb_root) / "openvla-runs"
    pointer = run_root / "last_run.json"
    candidates: list[Path] = []
    if pointer.is_file():
        try:
            info = json.loads(pointer.read_text())
            run_dir = info.get("run_dir")
            if run_dir:
                candidates.append(Path(run_dir))
        except (json.JSONDecodeError, OSError):
            pass
    candidates.append(run_root / "openvla-7b-finetuned-raccoonbot")

    for cand in candidates:
        if (cand / "config.json").is_file():
            return str(cand.resolve())

    tried = "\n  - ".join(str(c) for c in candidates)
    raise RuntimeError(
        "merged checkpoint not found (no run_dir with config.json).\n"
        f"checked:\n  - {tried}\n"
        "run 050_finetune then 055_merge_lora, or pass --model_path. "
        f"(run_root={run_root})"
    )


def tfds_data_dir(rb_root: Path) -> Path:
    return Path(rb_root) / "tensorflow_datasets"


def raw_demos_dir(rb_root: Path) -> Path:
    return mujoco_dir(rb_root) / "raccoon_grasp_colored_cylinder"


def stack_demos_dir(rb_root: Path) -> Path:
    return mujoco_dir(rb_root) / "raccoon_pick_and_stack"
