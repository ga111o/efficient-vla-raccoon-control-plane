import argparse
import json
import os
import shlex
import shutil
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
    print(f"[085] server health wait: {health} (timeout={timeout}s)", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"policy server exited during startup (exit={proc.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    print("[085] server ready.", flush=True)
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


def _resolve_num_workers(spec, num_episodes):
    cap = max(1, num_episodes)
    if str(spec).lower() == "auto":
        cores = os.cpu_count() or 1
        return max(1, min(cores, cap))
    try:
        n = int(spec)
    except ValueError:
        raise SystemExit(f"[085] --num-workers must be int or 'auto': {spec!r}")
    if n < 1:
        raise SystemExit("[085] --num-workers must be >= 1")
    return min(n, cap)


def _split_counts(total, parts):
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _run_workers_parallel(cmds, cwd, env):
    procs = []
    failures = []
    for i, cmd in enumerate(cmds):
        printable = " ".join(shlex.quote(str(c)) for c in cmd)
        print(f"[085] worker {i}: {printable}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=env))
    for i, pr in enumerate(procs):
        pr.wait()
        if pr.returncode != 0:
            failures.append((i, pr.returncode))
    if failures:
        detail = ", ".join(f"#{i}(exit={rc})" for i, rc in failures)
        raise RuntimeError(f"rollout worker failure: {detail}")


def _merge_reports(part_paths, report_path):
    parts = []
    for pp in part_paths:
        p = Path(pp)
        if not p.is_file():
            raise RuntimeError(f"partial report missing (worker may have failed): {p}")
        parts.append(json.loads(p.read_text(encoding="utf-8")))

    if not parts:
        raise RuntimeError("no partial reports to merge")

    def _wmean(pairs):
        num = den = 0.0
        for v, w in pairs:
            if v is not None and w:
                num += v * w
                den += w
        return (num / den) if den else None

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    first = parts[0]
    colors = list(first.get("per_color", {}).keys())
    for pr in parts[1:]:
        for c in pr.get("per_color", {}):
            if c not in colors:
                colors.append(c)

    per_color = {}
    total_eps = total_succ = 0
    overall_step_pairs = []
    for c in colors:
        eps = succ = 0
        step_pairs = []
        for pr in parts:
            st = pr.get("per_color", {}).get(c)
            if not st:
                continue
            n = st.get("episodes", 0)
            s = st.get("successes", 0)
            eps += n
            succ += s
            avg_steps = st.get("avg_steps_to_success")
            if avg_steps is not None and s:
                step_pairs.append((avg_steps, s))
        avg_steps = _wmean(step_pairs)
        per_color[c] = {
            "episodes": eps,
            "successes": succ,
            "success_rate": (succ / eps) if eps else 0.0,
            "avg_steps_to_success": avg_steps,
        }
        total_eps += eps
        total_succ += succ
        if avg_steps is not None and succ:
            overall_step_pairs.append((avg_steps, succ))

    per_episode = []
    for pr in parts:
        per_episode.extend(pr.get("diagnostics", {}).get("per_episode", []))
    per_episode.sort(key=lambda d: d.get("episode", 0))

    lat_pairs = [(pr.get("avg_inference_latency_s"), pr.get("total_inference_calls", 0))
                 for pr in parts]

    report = {
        "model_path": first.get("model_path"),
        "task": first.get("task"),
        "unnorm_key": first.get("unnorm_key"),
        "num_episodes": total_eps,
        "max_steps": first.get("max_steps"),
        "overall": {
            "successes": total_succ,
            "success_rate": (total_succ / total_eps) if total_eps else 0.0,
            "avg_steps_to_success": _wmean(overall_step_pairs),
        },
        "per_color": per_color,
        "avg_inference_latency_s": _wmean(lat_pairs),
        "total_inference_calls": sum(pr.get("total_inference_calls", 0) for pr in parts),
        "hz": first.get("hz"),
        "max_delta_xyz": first.get("max_delta_xyz"),
        "diagnostics": {
            "per_episode": per_episode,
            "avg_clip_hit_rate": _mean([d.get("clip_hit_rate") for d in per_episode]),
            "total_gripper_close_steps": sum(
                d.get("gripper_close_steps", 0) for d in per_episode),
            "total_ik_fail_steps": sum(d.get("ik_fail_steps", 0) for d in per_episode),
            "total_ik_retry_steps": sum(d.get("ik_retry_steps", 0) for d in per_episode),
            "avg_ee_path_len": _mean([d.get("ee_path_len") for d in per_episode]),
        },
        "num_workers": len(parts),
    }

    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[085] merged {len(parts)} worker reports -> {rp} "
          f"(num_episodes={total_eps}, successes={total_succ})", flush=True)


_BAR_COLORS = {"red": "#d62728", "blue": "#1f77b4",
               "green": "#2ca02c", "yellow": "#e6b800"}


def _visualize_report(report_path, vis_path):
    rp = Path(report_path)
    if not rp.is_file():
        print(f"[085] report missing, skip visualization: {rp}", flush=True)
        return
    report = json.loads(rp.read_text(encoding="utf-8"))

    per_color = report.get("per_color", {})
    if not per_color:
        print(f"[085] per_color empty, skip visualization: {rp}", flush=True)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = list(per_color.keys())
    palette = [_BAR_COLORS.get(c, "#7f7f7f") for c in colors]
    x = range(len(colors))

    rates = [per_color[c].get("success_rate") or 0.0 for c in colors]
    counts = [(per_color[c].get("successes", 0), per_color[c].get("episodes", 0))
              for c in colors]
    steps = [per_color[c].get("avg_steps_to_success") for c in colors]

    overall = report.get("overall", {})
    overall_rate = overall.get("success_rate") or 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(x, rates, color=palette, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(colors)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("success rate")
    ax1.set_title("Success rate per color")
    for rect, (s, n) in zip(bars, counts):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                 f"{s}/{n}", ha="center", va="bottom", fontsize=9)
    ax1.axhline(overall_rate, color="black", linestyle="--", linewidth=1,
                label=f"overall {overall_rate:.2f}")
    ax1.legend(loc="upper right")

    step_vals = [s if s is not None else 0 for s in steps]
    bars2 = ax2.bar(x, step_vals, color=palette, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(colors)
    top = max([s for s in steps if s is not None], default=1.0) * 1.15
    ax2.set_ylim(0, top)
    ax2.set_ylabel("avg steps to success")
    ax2.set_title("Avg steps to success per color (successes only)")
    for rect, s in zip(bars2, steps):
        label = f"{s:.1f}" if s is not None else "-"
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + top * 0.02,
                 label, ha="center", va="bottom", fontsize=9)

    model_name = Path(str(report.get("model_path", "N/A"))).name
    lat = report.get("avg_inference_latency_s")
    lat_str = f"{lat * 1000:.0f} ms/call" if lat is not None else "latency N/A"
    ov_steps = overall.get("avg_steps_to_success")
    ov_steps_str = f"{ov_steps:.1f}" if ov_steps is not None else "N/A"
    succ = overall.get("successes", 0)
    n_eps = report.get("num_episodes", 0)
    fig.suptitle(
        f"[{report.get('task', '?')}] closed-loop eval   |   "
        f"overall {succ}/{n_eps} ({overall_rate:.0%})   |   "
        f"avg steps {ov_steps_str}   |   {lat_str}   |   model: {model_name}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    vis = Path(vis_path)
    vis.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[085] visualization saved: {vis}", flush=True)


def _sample_indices(n_total, n_show):
    if n_show >= n_total:
        return list(range(n_total))
    if n_show <= 1:
        return [n_total - 1]
    return [round(i * (n_total - 1) / (n_show - 1)) for i in range(n_show)]


def _build_frame_montages(rollouts_dir, n_frames):
    rd = Path(rollouts_dir)
    ep_dirs = sorted(d for d in rd.iterdir()
                     if d.is_dir() and d.name.startswith("ep_")) if rd.is_dir() else []
    if not ep_dirs:
        print(f"[085] no rollout frames (ep_*/), skip montages: {rd}", flush=True)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = rd / "montages"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_made = 0
    for ep in ep_dirs:
        frames = sorted(ep.glob("frame_*.png"))
        if not frames:
            continue

        meta = {}
        mp = ep / "meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        idxs = _sample_indices(len(frames), n_frames)
        sel = [frames[i] for i in idxs]

        fig, axes = plt.subplots(1, len(sel), figsize=(3 * len(sel), 3))
        if len(sel) == 1:
            axes = [axes]
        for ax, i, fp in zip(axes, idxs, sel):
            ax.imshow(Image.open(fp))
            ax.set_title(f"step {i}", fontsize=10)
            ax.axis("off")

        instruction = meta.get("instruction", meta.get("language_instruction", "N/A"))
        color = meta.get("target_color", "N/A")
        success = meta.get("success", "N/A")
        steps = meta.get("steps_to_success")
        steps_str = steps if steps is not None else "-"
        fig.suptitle(
            f"{ep.name}   |   \"{instruction}\"\n"
            f"color={color}   success={success}   "
            f"steps_to_success={steps_str}   (frames={len(frames)})",
            fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

        out_path = out_dir / f"{ep.name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_made += 1

    print(f"[085] frame montages saved: {n_made} -> {out_dir}", flush=True)


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
    p.add_argument("--base-body", type=str, default="stack_base")
    p.add_argument("--base-label", type=str, default="base")
    p.add_argument("--num-episodes", type=int, default=4)
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--hz", type=int, default=10)
    p.add_argument("--max-delta-xyz", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-path", type=str, default=None)
    p.add_argument("--render-video", type=str, default=None)
    p.add_argument("--report-vis-path", type=str, default=None)
    p.add_argument("--no-report-vis", action="store_true")
    p.add_argument("--frame-montage-frames", type=int, default=6)
    p.add_argument("--no-frame-montage", action="store_true")
    p.add_argument("--vis-only", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    env = setup_env(rb)
    ov = openvla_dir(rb)
    mj = mujoco_dir(rb)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ov), str(mj), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    report_path = args.report_path or str(
        Path(__file__).resolve().parent / "outputs" / "eval" / "eval_stack_report.json")
    vis_path = args.report_vis_path or str(Path(report_path).with_suffix(".png"))

    if args.render_video:
        render_dir = args.render_video
    elif not args.no_frame_montage:
        render_dir = str(Path(report_path).parent / "stack_rollouts")
    else:
        render_dir = None

    if args.vis_only:
        if not args.no_report_vis:
            _visualize_report(report_path, vis_path)
        if not args.no_frame_montage and render_dir:
            _build_frame_montages(render_dir, args.frame_montage_frames)
        return

    model_path = resolve_model_path(rb, args.model_path)
    xml_path = args.xml_path or str(mj / "Raccoon_stack_scene.xml")
    server_url = f"http://{args.host}:{args.port}"

    server_cmd = [
        sys.executable, SERVER,
        "--model_path", model_path,
        "--default-unnorm-key", args.default_unnorm_key,
        "--device", args.device,
        "--host", args.host,
        "--port", str(args.port),
    ]
    print(f"\n[085] starting policy server (cwd={ov})\n      $ {' '.join(server_cmd)}\n",
          flush=True)
    server = subprocess.Popen(server_cmd, cwd=str(ov), env=env)

    try:
        _wait_health(server_url, server, args.server_startup_timeout)

        num_workers = _resolve_num_workers(args.num_workers, args.num_episodes)
        if str(args.num_workers).lower() == "auto":
            print(f"[085] --num-workers auto -> {num_workers} "
                  f"(cpu_count={os.cpu_count()}, num_episodes={args.num_episodes})",
                  flush=True)

        def _worker_cmd(n_eps, offset, seed, rpath):
            cmd = [
                sys.executable, WORKER,
                "--server-url", server_url,
                "--model_path", model_path,
                "--default-unnorm-key", args.default_unnorm_key,
                "--task", "stack",
                "--base-body", args.base_body,
                "--base-label", args.base_label,
                "--xml-path", xml_path,
                "--num-episodes", str(n_eps),
                "--episode-index-offset", str(offset),
                "--max-steps", str(args.max_steps),
                "--hz", str(args.hz),
                "--max-delta-xyz", str(args.max_delta_xyz),
                "--seed", str(seed),
                "--report-path", rpath,
            ]
            if render_dir:
                cmd += ["--render-video", render_dir]
            return cmd

        if num_workers == 1:
            run(_worker_cmd(args.num_episodes, 0, args.seed, report_path),
                cwd=ov, extra_env=env)
        else:
            counts = _split_counts(args.num_episodes, num_workers)
            parts_dir = Path(report_path).parent / "_stack_eval_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            cmds, part_paths, offset = [], [], 0
            for i, n_eps in enumerate(counts):
                if n_eps <= 0:
                    continue
                rpath = str(parts_dir / f"part_{i:02d}.json")
                cmds.append(_worker_cmd(n_eps, offset, args.seed + i, rpath))
                part_paths.append(rpath)
                offset += n_eps
            _run_workers_parallel(cmds, cwd=ov, env=env)
            _merge_reports(part_paths, report_path)
            shutil.rmtree(parts_dir, ignore_errors=True)

        if not args.no_report_vis:
            _visualize_report(report_path, vis_path)
        if not args.no_frame_montage and render_dir:
            _build_frame_montages(render_dir, args.frame_montage_frames)
    finally:
        print("[085] stopping policy server.", flush=True)
        _terminate(server)


if __name__ == "__main__":
    main()
