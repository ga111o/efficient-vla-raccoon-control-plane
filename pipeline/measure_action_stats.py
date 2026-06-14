import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, raccoon_dataset_dir  # noqa: E402


def quantile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[int((len(s) - 1) * p)]


def collect_episode_stats(episode_json_path):
    steps = json.loads(Path(episode_json_path).read_text())["steps"]
    norms = [math.hypot(a[0], a[1], a[2]) for a in (st["action"] for st in steps)]
    grip = [float(st["action"][6]) >= 0.5 for st in steps]
    close_tr = sum(1 for a, b in zip(grip, grip[1:]) if not a and b)
    open_tr = sum(1 for a, b in zip(grip, grip[1:]) if a and not b)
    closed_frac = (sum(grip) / len(grip)) if grip else 0.0
    return len(steps), norms, close_tr, open_tr, closed_frac


def load_checkpoint_q99(stats_path, unnorm_key):
    info = json.loads(Path(stats_path).read_text())
    node = info.get(unnorm_key) or (info if "action" in info else None)
    if node is None and len(info) == 1:
        node = next(iter(info.values()))
    if not node or "action" not in node or "q99" not in node["action"]:
        return None
    return [float(v) for v in node["action"]["q99"][:3]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--splits", type=str, default="train")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--current-max-delta-xyz", type=float, default=0.05)
    p.add_argument("--current-max-steps", type=int, default=150)
    p.add_argument("--dataset-statistics", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="raccoon_pick_place")
    p.add_argument("--report-path", type=str, default=None)
    args = p.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = raccoon_dataset_dir(resolve_rb_root()) / "openvla_rlds_intermediate"
    if not data_root.is_dir():
        raise SystemExit(
            f"[measure] data root not found: {data_root}\n"
            "run on server with data or pass --data-root")

    episode_paths = []
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        episode_paths += sorted((data_root / split).glob("episode_*/episode.json"))
    if not episode_paths:
        raise SystemExit(f"[measure] no episode.json under {data_root}/{args.splits}")

    random.seed(args.seed)
    random.shuffle(episode_paths)
    if args.sample > 0:
        episode_paths = episode_paths[:args.sample]

    all_norms = []
    steps_per_ep = []
    eps_with_close = eps_with_open = 0
    closed_fracs = []
    for ep_path in episode_paths:
        num_steps, norms, close_tr, open_tr, closed_frac = collect_episode_stats(ep_path)
        all_norms += norms
        steps_per_ep.append(num_steps)
        eps_with_close += int(close_tr >= 1)
        eps_with_open += int(open_tr >= 1)
        closed_fracs.append(closed_frac)

    n_ep = len(episode_paths)
    q99 = quantile(all_norms, 0.99)
    p99_steps = quantile(steps_per_ep, 0.99)
    rec_clip = math.ceil(q99 * 1.2 * 1000) / 1000

    report = {
        "data_root": str(data_root),
        "splits": args.splits,
        "episodes_measured": n_ep,
        "dxyz_norm": {
            "q99": round(q99, 4),
            "max": round(max(all_norms), 4),
            "frac_over_0.01": round(sum(n > 0.01 for n in all_norms) / len(all_norms), 4),
            f"frac_over_{args.current_max_delta_xyz}": round(
                sum(n > args.current_max_delta_xyz for n in all_norms) / len(all_norms), 4),
        },
        "steps_per_episode": {
            "p50": quantile(steps_per_ep, 0.50),
            "p99": p99_steps,
            "max": max(steps_per_ep),
        },
        "gripper": {
            "eps_with_close_transition": f"{eps_with_close}/{n_ep}",
            "eps_with_open_transition": f"{eps_with_open}/{n_ep}",
            "mean_closed_step_frac": round(sum(closed_fracs) / n_ep, 3),
        },
        "recommendations": {
            "max_delta_xyz": rec_clip,
            "max_steps_at_least": p99_steps,
        },
    }

    if args.dataset_statistics:
        ckpt_q99 = load_checkpoint_q99(args.dataset_statistics, args.unnorm_key)
        if ckpt_q99 is None:
            report["checkpoint_stats"] = {"error": "action q99 not found"}
        else:
            ratio = max(ckpt_q99) / q99 if q99 > 0 else None
            report["checkpoint_stats"] = {
                "q99_dxyz": [round(v, 4) for v in ckpt_q99],
                "ratio_vs_measured_q99": round(ratio, 3) if ratio else None,
                "verdict": ("OK" if ratio and 0.7 <= ratio <= 1.5 else "WARNING"),
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n[measure] --max-delta-xyz: q99x1.2 = {rec_clip} (current {args.current_max_delta_xyz})")
    flag = "raise" if args.current_max_steps < p99_steps else "ok"
    print(f"[measure] --max-steps: p99 = {p99_steps} (current {args.current_max_steps} -> {flag})")
    if eps_with_close < n_ep:
        print(f"[measure] warning: {n_ep - eps_with_close} episodes without close transition")

    if args.report_path:
        rp = Path(args.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] saved: {rp}")


if __name__ == "__main__":
    main()
