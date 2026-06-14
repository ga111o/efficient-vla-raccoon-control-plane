from __future__ import annotations

import argparse
import codecs
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TypedDict

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PIPELINE_DIR / "pipeline.toml"
RUNS_DIR = PIPELINE_DIR / "outputs" / "runs"


class Stage(TypedDict):
    id: str
    script: str
    enabled: bool
    args: list[str]


STAGES: list[Stage] = [
    {"id": "000", "script": "000_check_env.py", "enabled": True, "args": []},
    {"id": "010", "script": "010_generate_demos.py", "enabled": True,
     "args": ["--num-workers", "auto"]},
    {"id": "015", "script": "015_generate_stack_demos.py", "enabled": True,
     "args": ["--num-workers", "auto"]},
    {"id": "020", "script": "020_convert_rlds_intermediate.py", "enabled": True,
     "args": ["--val_ratio", "0.1", "--with-stack"]},
    {"id": "030", "script": "030_build_tfds.py", "enabled": True, "args": []},
    {"id": "040", "script": "040_visualize_episode.py", "enabled": True,
     "args": ["--episode-index", "0", "--num-frames", "5"]},
    {"id": "050", "script": "050_finetune.py", "enabled": True,
     "args": ["--max_steps", "100", "--save_steps", "100"]},
    {"id": "051", "script": "051_finetune_sandwich.py", "enabled": False,
     "args": ["--backbone-precision", "fp8", "--quant-backend", "torchao",
              "--batch_size", "4", "--grad_accumulation_steps", "4",
              "--max_steps", "11000", "--save_steps", "1000"]},
    {"id": "055", "script": "055_merge_lora.py", "enabled": True, "args": []},
    {"id": "060", "script": "060_download_checkpoint.py", "enabled": False, "args": []},
    {"id": "070", "script": "070_serve.py", "enabled": False,
     "args": ["--host", "0.0.0.0", "--port", "8000"]},
    {"id": "075", "script": "075_offline_infer.py", "enabled": True,
     "args": ["--mode", "eval", "--num-steps", "3"]},
    {"id": "080", "script": "080_evaluate.py", "enabled": True,
     "args": ["--num-episodes", "20", "--max-steps", "30"]},
    {"id": "085", "script": "085_evaluate_stack.py", "enabled": True,
     "args": ["--num-episodes", "4", "--max-steps", "150"]},
]


def _by_id() -> dict[str, Stage]:
    return {s["id"]: s for s in STAGES}


def apply_config(config_path: Path | None) -> Path | None:
    explicit = config_path is not None
    path = config_path or DEFAULT_CONFIG
    if not path.is_file():
        if explicit:
            raise SystemExit(f"[run_pipeline] config not found: {path}")
        return None
    if tomllib is None:
        _log("[run_pipeline] tomllib missing, using STAGES defaults")
        return None

    with path.open("rb") as f:
        data = tomllib.load(f)

    stage_cfg = data.get("stage", {})
    known = _by_id()
    unknown = [sid for sid in stage_cfg if sid not in known]
    if unknown:
        raise SystemExit(
            f"[run_pipeline] unknown stage ids: {', '.join(unknown)} "
            f"(available: {', '.join(known)})")

    for sid, cfg in stage_cfg.items():
        stage = known[sid]
        if "enabled" in cfg:
            stage["enabled"] = bool(cfg["enabled"])
        if "args" in cfg:
            stage["args"] = [str(a) for a in cfg["args"]]

    _log(f"[run_pipeline] loaded config: {path}")
    return path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str = "") -> None:
    prefix = _ts() + " "
    out = "\n".join((prefix + ln) if ln else ln for ln in msg.split("\n"))
    sys.stdout.write(out + "\n")
    sys.stdout.flush()


def _stream_with_timestamps(pipe: IO[bytes]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    out = sys.stdout
    fd = pipe.fileno()
    at_line_start = True
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if not text:
            continue
        parts: list[str] = []
        for ch in text:
            if at_line_start and ch not in ("\n", "\r"):
                parts.append(_ts() + " ")
                at_line_start = False
            parts.append(ch)
            if ch in ("\n", "\r"):
                at_line_start = True
        out.write("".join(parts))
        out.flush()
    tail = decoder.decode(b"", final=True)
    if tail:
        out.write((_ts() + " " if at_line_start else "") + tail)
        out.flush()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PIPELINE_DIR), capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _write_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_id_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [tok for tok in value.replace(",", " ").split() if tok]


def _validate_ids(ids: list[str]) -> None:
    known = _by_id()
    bad = [i for i in ids if i not in known]
    if bad:
        raise SystemExit(
            f"[run_pipeline] unknown stage id: {', '.join(bad)} "
            + f"(available: {', '.join(s['id'] for s in STAGES)})"
        )


def select_stages(args: argparse.Namespace) -> list[Stage]:
    only = _parse_id_list(args.only)
    skip = _parse_id_list(args.skip)
    enable = _parse_id_list(args.enable)
    _validate_ids(only + skip + enable + ([args.from_id] if args.from_id else [])
                  + ([args.to_id] if args.to_id else []))

    ids = [s["id"] for s in STAGES]
    start = ids.index(args.from_id) if args.from_id else 0
    end = ids.index(args.to_id) + 1 if args.to_id else len(STAGES)
    if start > end - 1:
        raise SystemExit(
            f"[run_pipeline] --from {args.from_id} after --to {args.to_id}")
    window = STAGES[start:end]

    selected = []
    for s in window:
        if only:
            if s["id"] in only:
                selected.append(s)
            continue
        if s["id"] in skip:
            continue
        if s["enabled"] or s["id"] in enable:
            selected.append(s)
    return selected


def build_cmd(stage: Stage) -> list[str]:
    script = PIPELINE_DIR / stage["script"]
    if not script.is_file():
        raise SystemExit(f"[run_pipeline] script not found: {script}")
    return [sys.executable, str(script), *stage["args"]]


_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _Interrupted(Exception):
    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def _terminate_group(proc: "subprocess.Popen[bytes]", grace: float = 10.0) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        return

    prev = {s: signal.signal(s, signal.SIG_IGN) for s in _SIGNALS}
    try:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            proc.poll()
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, OSError):
                break
            time.sleep(0.2)
        else:
            _log(f"[run_pipeline] group {pgid} did not exit in {grace:.0f}s, SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    finally:
        for s, h in prev.items():
            signal.signal(s, h)


def run_stage(cmd: list[str]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(PIPELINE_DIR), start_new_session=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if proc.stdout is not None:
            _stream_with_timestamps(proc.stdout)
        return proc.wait()
    finally:
        _terminate_group(proc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip", type=str, default=None)
    p.add_argument("--from", dest="from_id", type=str, default=None)
    p.add_argument("--to", dest="to_id", type=str, default=None)
    p.add_argument("--enable", type=str, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--no-manifest", action="store_true")
    args = p.parse_args()

    config_path = apply_config(Path(args.config).resolve() if args.config else None)

    if args.list:
        print("# pipeline stages")
        for s in STAGES:
            flag = "on " if s["enabled"] else "off"
            extra = " ".join(shlex.quote(a) for a in s["args"]) or "(no args)"
            print(f"  [{flag}] {s['id']}  {s['script']:<34} args: {extra}")
        return

    selected = select_stages(args)
    if not selected:
        raise SystemExit("[run_pipeline] no stages selected. use --list")

    _log("[run_pipeline] order: " + " -> ".join(s["id"] for s in selected))

    def _handler(signum, _frame):
        raise _Interrupted(signum)

    for _s in _SIGNALS:
        signal.signal(_s, _handler)

    run_stamp = _utc_stamp()
    stage_records: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "run_id": run_stamp,
        "start": datetime.now(timezone.utc).isoformat(),
        "end": None,
        "duration_s": None,
        "git_sha": _git_sha(),
        "config": str(config_path) if config_path else None,
        "dry_run": bool(args.dry_run),
        "selected_stages": [s["id"] for s in selected],
        "status": "running",
        "stages": stage_records,
    }
    manifest_path = RUNS_DIR / run_stamp / "manifest.json"
    run_start = time.monotonic()

    total = len(selected)
    had_failure = False
    try:
        for idx, stage in enumerate(selected, 1):
            cmd = build_cmd(stage)
            printable = " ".join(shlex.quote(c) for c in cmd)
            header = f"[{idx}/{total}] STAGE {stage['id']}  ({stage['script']})"
            _log(f"\n{'=' * 72}\n{header}\n  $ {printable}\n{'=' * 72}")

            if args.dry_run:
                stage_records.append({
                    "id": stage["id"], "script": stage["script"],
                    "args": list(stage["args"]), "cmd": printable,
                    "exit_code": None, "duration_s": None,
                    "start": None, "end": None,
                })
                continue

            stage_start_iso = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            returncode = run_stage(cmd)
            duration = time.monotonic() - t0
            stage_records.append({
                "id": stage["id"], "script": stage["script"],
                "args": list(stage["args"]), "cmd": printable,
                "exit_code": returncode, "duration_s": round(duration, 3),
                "start": stage_start_iso,
                "end": datetime.now(timezone.utc).isoformat(),
            })
            _log(f"[run_pipeline] STAGE {stage['id']} done "
                 f"(exit={returncode}, + {duration:.1f}s)")

            if returncode != 0:
                msg = (f"\n[run_pipeline] STAGE {stage['id']} failed "
                       f"(exit={returncode}) - {stage['script']}")
                if args.continue_on_error:
                    _log(msg + "  continue-on-error")
                    had_failure = True
                    continue
                _log(msg + "  stopping (fail-fast)")
                manifest["status"] = "failed"
                sys.exit(returncode)
    except _Interrupted as e:
        name = signal.Signals(e.signum).name
        _log(f"\n[run_pipeline] {name} received, cleaning up")
        manifest["status"] = "interrupted"
        sys.exit(128 + e.signum)
    else:
        if args.dry_run:
            manifest["status"] = "dry-run"
        else:
            manifest["status"] = "failed" if had_failure else "ok"
        _log(f"\n[run_pipeline] done: {' -> '.join(s['id'] for s in selected)}")
    finally:
        total_dur = time.monotonic() - run_start
        manifest["end"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_s"] = round(total_dur, 3)
        if not args.no_manifest:
            try:
                _write_manifest(manifest_path, manifest)
                _log(f"[run_pipeline] total {total_dur:.1f}s | manifest: {manifest_path}")
            except Exception as exc:
                _log(f"[run_pipeline] manifest write failed: {exc}")
        else:
            _log(f"[run_pipeline] total {total_dur:.1f}s (no manifest)")


if __name__ == "__main__":
    main()
