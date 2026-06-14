from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

PIPELINE_DIR = Path(__file__).resolve().parent
OUT_DIR = PIPELINE_DIR / "outputs"
STAGE_051 = PIPELINE_DIR / "051_finetune_sandwich.py"
STAGE_055 = PIPELINE_DIR / "055_merge_lora.py"
STAGE_085 = PIPELINE_DIR / "085_evaluate_stack.py"

CONFIGS: dict[str, list[str]] = {
    "A": [],
    "B": ["--lora_alpha", "32"],
    "C": ["--use_rslora"],
    "D": ["--lora_rank", "64", "--use_rslora",
          "--batch_size", "2", "--grad_accumulation_steps", "8"],
}

_STEP_RE = re.compile(r"\[step \d+\] loss=([0-9.]+) .*?peak_alloc=([0-9.]+)GB")
_VAL_RE = re.compile(r"\[val step (\d+)\] loss=([0-9.]+) acc=[0-9.]+ l1=([0-9.]+)")


def _print_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _tee_run(cmd: list[str], log_path: Path | None) -> int:
    print(f"\n[sweep] $ {_print_cmd(cmd)}\n", flush=True)
    env = {**_child_env()}
    f = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(PIPELINE_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if f:
                f.write(line)
        return proc.wait()
    finally:
        if f:
            f.close()


def _child_env() -> dict[str, str]:
    import os
    return {**os.environ, "PYTHONUNBUFFERED": "1"}


def _build_051_cmd(name: str, steps: int, save_steps: int, lr: float) -> list[str]:
    base = [
        sys.executable, str(STAGE_051),
        "--backbone-precision", "fp8",
        "--quant-backend", "torchao",
        "--vision-precision", "bf16",
        "--batch_size", "4",
        "--grad_accumulation_steps", "4",
        "--learning_rate", str(lr),
        "--max_steps", str(steps),
        "--save_steps", str(save_steps),
        "--run_id_note", f"sweep-{name}",
    ]
    return base + CONFIGS[name]


def _build_055_cmd() -> list[str]:
    return [sys.executable, str(STAGE_055)]


def _build_085_cmd(report_path: Path, episodes: int,
                   max_steps: int, hz: int, max_delta_xyz: float) -> list[str]:
    return [
        sys.executable, str(STAGE_085),
        "--num-episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--hz", str(hz),
        "--max-delta-xyz", str(max_delta_xyz),
        "--num-workers", "auto",
        "--report-path", str(report_path),
    ]


def _parse_train_log(path: Path) -> dict[str, object]:
    train_loss_min: float | None = None
    peak_alloc_max: float | None = None
    val_l1_best: float | None = None
    val_l1_best_step: int | None = None
    val_loss_at_best: float | None = None
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _STEP_RE.search(line)
            if m:
                loss, pa = float(m.group(1)), float(m.group(2))
                if train_loss_min is None or loss < train_loss_min:
                    train_loss_min = loss
                if peak_alloc_max is None or pa > peak_alloc_max:
                    peak_alloc_max = pa
            mv = _VAL_RE.search(line)
            if mv:
                step, vloss, vl1 = int(mv.group(1)), float(mv.group(2)), float(mv.group(3))
                if val_l1_best is None or vl1 < val_l1_best:
                    val_l1_best, val_l1_best_step, val_loss_at_best = vl1, step, vloss
    return {
        "train_loss_min": train_loss_min, "peak_alloc_max_gb": peak_alloc_max,
        "val_l1_best": val_l1_best, "val_l1_best_step": val_l1_best_step,
        "val_loss_at_best": val_loss_at_best,
    }


def _parse_eval(report_path: Path) -> dict[str, object]:
    empty = {
        "success_rate": None, "successes": None, "eval_episodes": None,
        "avg_min_xy_dist": None, "avg_first_close_ee_z": None, "avg_cube_disp_xy": None,
    }
    try:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    overall = rep.get("overall", {})
    per_ep = rep.get("diagnostics", {}).get("per_episode", []) or []

    def _mean(key: str):
        xs = [e[key] for e in per_ep
              if isinstance(e, dict) and e.get(key) is not None]
        return (sum(xs) / len(xs)) if xs else None

    return {
        "success_rate": overall.get("success_rate"),
        "successes": overall.get("successes"),
        "eval_episodes": rep.get("num_episodes"),
        "avg_min_xy_dist": _mean("min_xy_dist"),
        "avg_first_close_ee_z": _mean("first_close_ee_z"),
        "avg_cube_disp_xy": _mean("cube_disp_xy"),
    }


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return "-"
    if spec:
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def _write_results(json_path: Path, md_path: Path,
                   rows: list[dict[str, object]], meta: dict[str, object]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"meta": meta, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    header = ("| config | extra args | steps | train_loss_min | val_l1_best (step) | "
              "success | avg_min_xy_dist | peak_alloc(GB) | status | dur(s) |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines = [
        f"# Capacity sweep - {meta.get('timestamp', '')}",
        "",
        f"- proxy_steps={meta.get('steps')} save_steps={meta.get('save_steps')} "
        f"lr={meta.get('learning_rate')} eval_episodes={meta.get('eval_episodes')}",
        "",
        header, sep,
    ]
    for r in rows:
        extra = " ".join(cast("list[str]", r.get("extra_args") or [])) or "(baseline)"
        succ = (f"{_fmt(r.get('success_rate'), '.2f')} "
                f"({_fmt(r.get('successes'))}/{_fmt(r.get('eval_episodes'))})")
        vl1 = f"{_fmt(r.get('val_l1_best'), '.4f')} ({_fmt(r.get('val_l1_best_step'))})"
        lines.append(
            f"| {r['config']} | {extra} | {_fmt(r.get('steps'))} | "
            f"{_fmt(r.get('train_loss_min'), '.4f')} | {vl1} | {succ} | "
            f"{_fmt(r.get('avg_min_xy_dist'), '.4f')} | "
            f"{_fmt(r.get('peak_alloc_max_gb'), '.2f')} | {r.get('status', '?')} | "
            f"{_fmt(r.get('duration_s'), '.0f')} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_configs(spec: str) -> list[str]:
    names = [t for t in spec.replace(",", " ").split() if t]
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        raise SystemExit(
            f"[sweep] unknown config: {', '.join(bad)} "
            f"(available: {', '.join(CONFIGS)})")
    return names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, default="B,C")
    p.add_argument("--proxy-steps", type=int, default=1200)
    p.add_argument("--final-steps", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--proxy-episodes", type=int, default=12)
    p.add_argument("--eval-max-steps", type=int, default=321)
    p.add_argument("--eval-hz", type=int, default=10)
    p.add_argument("--eval-max-delta-xyz", type=float, default=0.05)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    names = _parse_configs(args.configs)
    steps = args.final_steps if args.final_steps > 0 else args.proxy_steps
    save_steps = args.save_steps if args.save_steps > 0 else (
        1000 if args.final_steps > 0 else 500)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    sweep_dir = OUT_DIR / f"sweep_{ts}"
    json_path = OUT_DIR / f"sweep_{ts}.json"
    md_path = OUT_DIR / f"sweep_{ts}.md"
    meta: dict[str, object] = {
        "timestamp": ts, "configs": names, "steps": steps, "save_steps": save_steps,
        "learning_rate": args.learning_rate, "eval_episodes": args.proxy_episodes,
        "final": args.final_steps > 0, "skip_eval": bool(args.skip_eval),
    }

    print(f"[sweep] configs={names} steps={steps} save_steps={save_steps} "
          f"lr={args.learning_rate} eval_episodes={args.proxy_episodes} "
          f"{'(FINAL)' if args.final_steps > 0 else '(proxy)'}", flush=True)
    print(f"[sweep] output -> {json_path}\n               {md_path}\n               {sweep_dir}/",
          flush=True)

    if args.dry_run:
        for name in names:
            cfg_dir = sweep_dir / name
            report = cfg_dir / "eval_stack_report.json"
            print(f"\n--- config {name} ({' '.join(CONFIGS[name]) or 'baseline'}) ---")
            print("  051: " + _print_cmd(_build_051_cmd(name, steps, save_steps, args.learning_rate)))
            print("  055: " + _print_cmd(_build_055_cmd()))
            if not args.skip_eval:
                print("  085: " + _print_cmd(_build_085_cmd(
                    report, args.proxy_episodes, args.eval_max_steps,
                    args.eval_hz, args.eval_max_delta_xyz)))
        print("\n[sweep] dry-run only")
        return

    rows: list[dict[str, object]] = []
    for name in names:
        cfg_dir = sweep_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        train_log = cfg_dir / "train.log"
        report_path = cfg_dir / "eval_stack_report.json"
        row: dict[str, object] = {
            "config": name, "extra_args": CONFIGS[name], "steps": steps,
            "status": "running", "duration_s": None,
            "train_log": str(train_log), "eval_report": str(report_path),
        }
        rows.append(row)
        _write_results(json_path, md_path, rows, meta)

        t0 = time.monotonic()
        print(f"\n{'=' * 72}\n[sweep] CONFIG {name}  "
              f"({' '.join(CONFIGS[name]) or 'baseline'})\n{'=' * 72}", flush=True)
        try:
            rc = _tee_run(_build_051_cmd(name, steps, save_steps, args.learning_rate), train_log)
            if rc != 0:
                raise RuntimeError(f"051 failed (exit={rc})")
            row.update(_parse_train_log(train_log))

            rc = _tee_run(_build_055_cmd(), None)
            if rc != 0:
                raise RuntimeError(f"055 failed (exit={rc})")

            if not args.skip_eval:
                rc = _tee_run(_build_085_cmd(
                    report_path, args.proxy_episodes, args.eval_max_steps,
                    args.eval_hz, args.eval_max_delta_xyz), None)
                if rc != 0:
                    raise RuntimeError(f"085 failed (exit={rc})")
                row.update(_parse_eval(report_path))

            row["status"] = "ok"
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            if train_log.is_file():
                for k, v in _parse_train_log(train_log).items():
                    row.setdefault(k, v)
            print(f"\n[sweep] CONFIG {name} failed: {e}", flush=True)
            if args.stop_on_error:
                row["duration_s"] = round(time.monotonic() - t0, 1)
                _write_results(json_path, md_path, rows, meta)
                raise SystemExit(f"[sweep] stop-on-error at {name}")
        finally:
            row["duration_s"] = round(time.monotonic() - t0, 1)
            _write_results(json_path, md_path, rows, meta)
            print(f"[sweep] CONFIG {name} done (status={row['status']}, "
                  f"+ {row['duration_s']:.0f}s) - "
                  f"train_loss_min={_fmt(row.get('train_loss_min'), '.4f')} "
                  f"val_l1_best={_fmt(row.get('val_l1_best'), '.4f')} "
                  f"success_rate={_fmt(row.get('success_rate'), '.2f')}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n[sweep] done: {len(ok)}/{len(rows)} configs ok")
    print(f"[sweep] results: {md_path}")
    if ok:
        ranked = sorted(
            (r for r in ok if r.get("val_l1_best") is not None),
            key=lambda r: cast(float, r["val_l1_best"]))
        if ranked:
            w = ranked[0]
            print(f"[sweep] best val_l1: {w['config']} "
                  f"(val_l1={w['val_l1_best']:.4f}, train_loss_min="
                  f"{_fmt(w.get('train_loss_min'), '.4f')})")


if __name__ == "__main__":
    main()
