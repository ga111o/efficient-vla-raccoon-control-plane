import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import base64
import io
import json
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def predict_http(server_url, instruction, image_b64, unnorm_key, do_sample=False,
                 timeout=120):
    body = json.dumps({
        "instruction": instruction,
        "image_b64": image_b64,
        "unnorm_key": unnorm_key,
        "do_sample": do_sample,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/predict", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _add_mujoco_to_path(xml_path: str) -> Path:
    rb_root = os.environ.get("RB_ROOT")
    if rb_root:
        mj = Path(rb_root) / "Mujoco"
    else:
        mj = Path(xml_path).resolve().parent
    sys.path.insert(0, str(mj))
    return mj


def _img_to_b64(image_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _attach_delta_execution(rc) -> None:
    from raccoon_env import SyncSimRaccoonEnv

    rc.execute_delta_action7 = types.MethodType(
        SyncSimRaccoonEnv.execute_delta_action7, rc)


def _make_success_check(rc, args, target_body_name):
    if args.task == "stack":
        return lambda: rc.is_stack_success(
            src_body_name=target_body_name,
            base_body_name=args.base_body,
            touch_threshold=args.touch_threshold,
        )
    return lambda: rc.is_target_grasp_success(
        target_body_name=target_body_name,
        touch_threshold=args.touch_threshold,
    )


def rollout_episode(rc, server_url, args, target_color, object_specs, target_body_name):
    instruction = args.instruction_template.format(color=target_color, base=args.base_label)

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()
    if args.initial_settle_seconds > 0:
        rc.settle_steps(seconds=args.initial_settle_seconds)

    is_success = _make_success_check(rc, args, target_body_name)

    dt = 1.0 / args.hz
    latencies: list[float] = []
    frames: list[np.ndarray] = []

    cube_init = rc.get_object_pose(target_body_name)

    diag = {
        "steps_executed": 0,
        "clip_hit_steps": 0,
        "gripper_close_steps": 0,
        "ik_retry_steps": 0,
        "ik_fail_steps": 0,
        "ee_path_len": 0.0,
        "first_close_step": -1,
        "first_close_ee_z": -1.0,
        "first_close_xy_dist": -1.0,
        "min_xy_dist": -1.0,
        "min_xy_dist_step": -1,
        "min_xy_dist_ee_z": -1.0,
        "cube_disp_xy": -1.0,
    }
    min_xy_dist = float("inf")

    def _finalize():
        cube_now = rc.get_object_pose(target_body_name)
        if min_xy_dist != float("inf"):
            diag["min_xy_dist"] = round(min_xy_dist, 4)
        diag["cube_disp_xy"] = round(float(np.hypot(
            float(cube_now[0]) - float(cube_init[0]),
            float(cube_now[1]) - float(cube_init[1]))), 4)

    for step in range(args.max_steps):
        obs = rc.get_observation()
        if args.render_video:
            frames.append(obs["image"])

        t0 = time.perf_counter()
        out = predict_http(
            server_url,
            instruction=instruction,
            image_b64=_img_to_b64(obs["image"]),
            unnorm_key=args.default_unnorm_key,
            do_sample=args.do_sample,
        )
        latencies.append(time.perf_counter() - t0)

        action = out["action"]
        executed = False
        gripper_cmd = 0.0
        try:
            info = rc.execute_delta_action7(action, max_delta_xyz=args.max_delta_xyz)
            executed = True
            diag["steps_executed"] += 1
            raw = info["raw_delta_xyz"]
            raw_norm = float(np.linalg.norm(raw))
            if raw_norm > args.max_delta_xyz:
                diag["clip_hit_steps"] += 1
            gripper_cmd = float(info["gripper_cmd"])
            if gripper_cmd >= 0.5:
                diag["gripper_close_steps"] += 1
            if int(info["retry_count"]) > 0:
                diag["ik_retry_steps"] += 1
            diag["ee_path_len"] += float(np.linalg.norm(info["actual_move_xyz"]))
        except Exception as exc:
            diag["ik_fail_steps"] += 1
            print(f"  [step {step:02d}] execute_delta_action7 skip: {exc}", flush=True)

        rc.settle_steps(seconds=dt)

        if executed:
            ee_x, ee_y, ee_z = rc.get_ee_pose()
            cube = rc.get_object_pose(target_body_name)
            xy_dist = float(np.hypot(ee_x - float(cube[0]), ee_y - float(cube[1])))
            if xy_dist < min_xy_dist:
                min_xy_dist = xy_dist
                diag["min_xy_dist_step"] = step
                diag["min_xy_dist_ee_z"] = round(ee_z, 4)
            if gripper_cmd >= 0.5 and diag["first_close_step"] < 0:
                diag["first_close_step"] = step
                diag["first_close_ee_z"] = round(ee_z, 4)
                diag["first_close_xy_dist"] = round(xy_dist, 4)

        if is_success():
            _finalize()
            if args.render_video:
                frames.append(rc.get_observation()["image"])
                _save_frames(frames, args, target_color, True, step + 1, instruction, diag)
            return True, step + 1, latencies, diag

    _finalize()
    if args.render_video:
        _save_frames(frames, args, target_color, False, None, instruction, diag)
    return False, None, latencies, diag


def _save_frames(frames, args, target_color, success, steps, instruction, diag=None) -> None:
    out_dir = Path(args.render_video)
    idx = getattr(args, "_video_idx", 0)
    args._video_idx = idx + 1
    stem = f"ep_{idx:03d}_{target_color}"
    ep_dir = out_dir / stem
    ep_dir.mkdir(parents=True, exist_ok=True)

    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(ep_dir / f"frame_{i:04d}.png")

    meta = {
        "task": args.task,
        "instruction": instruction,
        "target_color": target_color,
        "success": bool(success),
        "steps_to_success": steps,
        "num_frames": len(frames),
    }
    if diag is not None:
        ns = max(diag.get("steps_executed", 0), 1)
        meta["diagnostics"] = {
            **diag,
            "clip_hit_rate": diag.get("clip_hit_steps", 0) / ns,
            "ee_path_len": round(diag.get("ee_path_len", 0.0), 4),
        }
    (ep_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8")

    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(out_dir / f"{stem}.mp4"), frames, fps=args.hz)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(
        description="OpenVLA closed-loop 평가 (MuJoCo 롤아웃 grasp 성공률)")
    p.add_argument("--server-url", type=str, required=True,
                   help="정책 추론 서버(openvla_server) 주소, 예: http://127.0.0.1:8000")
    p.add_argument("--model_path", type=str, default=None,
                   help="리포트 라벨용(추론은 --server-url 이 담당). 미지정 시 서버에 위임.")
    p.add_argument("--default-unnorm-key", type=str, default="raccoon_pick_place")
    p.add_argument("--task", type=str, default="grasp", choices=("grasp", "stack"),
                   help="평가 task. grasp(기본) 또는 stack(pick-and-stack).")
    p.add_argument("--base-body", type=str, default="stack_base",
                   help="stack task 의 베이스 큐브 body 이름 (기본: stack_base)")
    p.add_argument("--base-label", type=str, default="base",
                   help="stack instruction 의 {base} 라벨 (기본: 'base')")
    p.add_argument("--xml-path", type=str, default=None,
                   help="씬 XML (미지정 시 task 기본: grasp→Raccoon_colored_cylinder.xml, "
                        "stack→Raccoon_stack_scene.xml)")
    p.add_argument("--camera-name", type=str, default="front_view")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--episode-index-offset", type=int, default=0,
                   help="이 worker 가 맡은 에피소드의 전역 시작 인덱스. 색상 라운드로빈을 "
                        "전역적으로 균형 맞추고(=colors[(offset+ep)%len]), 프레임 저장 "
                        "stem(ep_XXX)을 전역 고유로 만들어 여러 worker 가 같은 "
                        "--render-video 디렉토리에 써도 충돌하지 않게 한다. "
                        "085 가 멀티코어 병렬 실행 시 worker 마다 다른 값을 준다.")
    p.add_argument("--max-steps", type=int, default=30,
                   help="에피소드당 최대 추론-실행 step (성공 시 조기 종료). "
                        "stack 은 carry 경로가 길어 ~150 권장.")
    p.add_argument("--hz", type=int, default=10,
                   help="step 당 시뮬레이션 진행 시간 dt=1/hz (수집과 동일 기본 10). "
                        "생성(015) hz 와 반드시 같은 값으로 둘 것 — 갈라지면 train↔rollout "
                        "타이밍 불일치로 grasp 가 전부 빗나간다.")
    p.add_argument("--max-delta-xyz", type=float, default=0.01,
                   help="execute_delta_action7 의 per-step xyz 클립(±m). 학습 델타 q99 "
                        "보다 작으면 정상 출력이 잘려 팔이 느려진다. 측정한 q99(×1.2) 로 둘 것.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--colors", type=str, default="red,blue,green,yellow",
                   help="평가 대상 색상 (쉼표 구분)")
    p.add_argument("--instruction-template", type=str, default=None,
                   help="instruction 템플릿. {color}/{base} 치환 (미지정 시 task 기본값)")
    p.add_argument("--initial-settle-seconds", type=float, default=0.3)
    p.add_argument("--touch-threshold", type=float, default=0.1)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--report-path", type=str, default=None,
                   help="결과 JSON 저장 경로 (미지정 시 stdout 만)")
    p.add_argument("--render-video", type=str, default=None,
                   help="(선택) 에피소드 프레임을 저장할 디렉토리. ep_XXX_<color>/frame_*.png "
                        "+ meta.json 으로 015 데이터셋과 동일 구조로 남기고(가능하면 mp4 도), "
                        "085 가 이 프레임으로 040 스타일 몽타주를 만든다.")
    args = p.parse_args()

    if args.task == "stack":
        if args.xml_path is None:
            args.xml_path = "Raccoon_stack_scene.xml"
        if args.instruction_template is None:
            args.instruction_template = "stack the {color} cube on the {base}"
    else:
        if args.xml_path is None:
            args.xml_path = "Raccoon_colored_cylinder.xml"
        if args.instruction_template is None:
            args.instruction_template = "grasp the {color} cylinder"

    mj_dir = _add_mujoco_to_path(args.xml_path)
    xml_path = Path(args.xml_path)
    if not xml_path.is_absolute():
        xml_path = mj_dir / xml_path
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    from raccoon_grasp_multicolor_scene_dataset import SyncSimRaccoonDataset

    colors = tuple(c.strip() for c in args.colors.split(",") if c.strip())

    rc = SyncSimRaccoonDataset(
        xml_path=str(xml_path),
        image_size=(256, 256),
        camera_name=args.camera_name,
        use_viewer=False,
    )
    _attach_delta_execution(rc)

    args._video_idx = args.episode_index_offset

    reserved_xy = rc.get_body_xy(args.base_body) if args.task == "stack" else None

    rng = np.random.default_rng(args.seed)

    per_color = {c: {"episodes": 0, "successes": 0, "steps_to_success": []}
                 for c in colors}
    all_latencies: list[float] = []
    episode_diags: list[dict] = []

    try:
        for ep in range(args.num_episodes):
            global_ep = args.episode_index_offset + ep
            target_color = colors[global_ep % len(colors)]
            object_specs = SyncSimRaccoonDataset.sample_object_specs(
                rng=rng, colors=colors, reserved_xy=reserved_xy)
            target_body_name = object_specs[target_color]["body_name"]

            success, steps, lats, diag = rollout_episode(
                rc, args.server_url, args, target_color, object_specs, target_body_name)

            per_color[target_color]["episodes"] += 1
            all_latencies.extend(lats)
            if success:
                per_color[target_color]["successes"] += 1
                per_color[target_color]["steps_to_success"].append(steps)

            ns = max(diag["steps_executed"], 1)
            ep_diag = {
                "episode": global_ep,
                "color": target_color,
                "success": bool(success),
                "steps_executed": diag["steps_executed"],
                "clip_hit_steps": diag["clip_hit_steps"],
                "clip_hit_rate": diag["clip_hit_steps"] / ns,
                "gripper_close_steps": diag["gripper_close_steps"],
                "ik_retry_steps": diag["ik_retry_steps"],
                "ik_fail_steps": diag["ik_fail_steps"],
                "ee_path_len": round(diag["ee_path_len"], 4),
                "first_close_step": diag["first_close_step"],
                "first_close_ee_z": diag["first_close_ee_z"],
                "first_close_xy_dist": diag["first_close_xy_dist"],
                "min_xy_dist": diag["min_xy_dist"],
                "min_xy_dist_ee_z": diag["min_xy_dist_ee_z"],
                "cube_disp_xy": diag["cube_disp_xy"],
            }
            episode_diags.append(ep_diag)

            print(f"[ep {ep:03d}] color={target_color:<6} success={success} "
                  f"steps={steps} | clip_hit_rate={ep_diag['clip_hit_rate']:.2f} "
                  f"gripper_close={diag['gripper_close_steps']} "
                  f"ik_fail={diag['ik_fail_steps']} "
                  f"ee_path={ep_diag['ee_path_len']:.3f}m | "
                  f"close@z={diag['first_close_ee_z']} xy={diag['first_close_xy_dist']} "
                  f"min_xy={diag['min_xy_dist']} cube_moved={diag['cube_disp_xy']}m",
                  flush=True)
    finally:
        rc.close()

    def _rate(s, n):
        return (s / n) if n else 0.0

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    color_report = {}
    total_eps = total_succ = 0
    all_steps: list[int] = []
    for c in colors:
        st = per_color[c]
        total_eps += st["episodes"]
        total_succ += st["successes"]
        all_steps.extend(st["steps_to_success"])
        color_report[c] = {
            "episodes": st["episodes"],
            "successes": st["successes"],
            "success_rate": _rate(st["successes"], st["episodes"]),
            "avg_steps_to_success": _mean(st["steps_to_success"]),
        }

    report = {
        "model_path": str(args.model_path),
        "task": args.task,
        "unnorm_key": args.default_unnorm_key,
        "num_episodes": total_eps,
        "max_steps": args.max_steps,
        "overall": {
            "successes": total_succ,
            "success_rate": _rate(total_succ, total_eps),
            "avg_steps_to_success": _mean(all_steps),
        },
        "per_color": color_report,
        "avg_inference_latency_s": _mean(all_latencies),
        "total_inference_calls": len(all_latencies),
        "hz": args.hz,
        "max_delta_xyz": args.max_delta_xyz,
        "diagnostics": {
            "per_episode": episode_diags,
            "avg_clip_hit_rate": _mean([d["clip_hit_rate"] for d in episode_diags]),
            "total_gripper_close_steps": sum(d["gripper_close_steps"] for d in episode_diags),
            "total_ik_fail_steps": sum(d["ik_fail_steps"] for d in episode_diags),
            "total_ik_retry_steps": sum(d["ik_retry_steps"] for d in episode_diags),
            "avg_ee_path_len": _mean([d["ee_path_len"] for d in episode_diags]),
            "avg_first_close_ee_z": _mean([d["first_close_ee_z"] for d in episode_diags
                                           if d["first_close_step"] >= 0]),
            "avg_first_close_xy_dist": _mean([d["first_close_xy_dist"] for d in episode_diags
                                              if d["first_close_step"] >= 0]),
            "avg_min_xy_dist": _mean([d["min_xy_dist"] for d in episode_diags
                                      if d["min_xy_dist"] >= 0]),
            "avg_cube_disp_xy": _mean([d["cube_disp_xy"] for d in episode_diags]),
            "episodes_with_close": sum(1 for d in episode_diags if d["first_close_step"] >= 0),
        },
    }

    print("\n===== CLOSED-LOOP EVAL REPORT =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report_path:
        rp = Path(args.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print(f"\n[OK] saved {rp}")


if __name__ == "__main__":
    main()
