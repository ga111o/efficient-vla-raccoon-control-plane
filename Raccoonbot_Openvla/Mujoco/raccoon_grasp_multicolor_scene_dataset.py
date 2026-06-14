import os
import json
import math
import shutil
import threading
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


class DatasetLogger:
    """
    Raw dataset logger.
    Saves:
      dataset_root/
        episode_000001/
          frame_000000.png
          frame_000001.png
          ...
          meta.json
    """
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(
        self,
        episode_id,
        instruction,
        goal_xy,
        box_init_xy,
        box_init_yaw,
        task_type="pick",
        target_color=None,
        target_body_name=None,
        all_object_init_poses=None,
        extra_meta=None,
    ):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            # grasp-only에서는 별도 place goal이 없으므로 초기 box 위치를 goal_xy로 둔다.
            # 기존 intermediate/RLDS 변환 코드와 호환되도록 2차원 필드는 유지한다.
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }

        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses
        # Task-specific extras (e.g. stack base body/top-z) live as free-form
        # meta keys only. The converter copies only known fields and the TFDS
        # builder ignores unknown ones, so adding these never changes the schema.
        if extra_meta is not None:
            for key, value in extra_meta.items():
                self.meta[key] = value

    def log_step(
        self,
        step_idx,
        image_rgb,
        joint_angles,
        gripper_state,
        object_pose,
        ee_pose,
        action,
        is_first=False,
        is_last=False,
    ):
        image_file = f"frame_{step_idx:06d}.png"
        image_path = self.episode_dir / image_file
        Image.fromarray(image_rgb).save(image_path)

        step_data = {
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        }
        self.meta["steps"].append(step_data)

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)

        meta_path = self.episode_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    """
    Synchronous MuJoCo dataset collector for RaccoonBot.

    Key design choices:
    - No background simulation thread
    - No real-time sleep-based settling
    - Main loop only: command -> run N mj_step -> render/save
    - Safe with viewer=False (physics still advances)
    """

    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0

    # Uploaded move_to code style uses centimeter-scale IK constants.
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    CYLINDER_BODY_BY_COLOR = {
        "red": "target_object",
        "blue": "target_object_blue",
        "green": "target_object_green",
        "yellow": "target_object_yellow",
    }
    CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

    # ---- pick-and-stack task constants ----------------------------------
    # Static base cube body in Raccoon_stack_scene.xml (world-fixed, no joint).
    STACK_BASE_BODY = "stack_base"
    # Colored cubes are 20mm tall (half-height 0.01); they rest on the floor
    # with their center at this height. Also the cube half-height used to compute
    # the target stacked height (base_top_z + CUBE_HALF_HEIGHT).
    CUBE_HALF_HEIGHT = 0.01
    # Waypoint heights reused by both grasp and stack plans (EE z, meters).
    GRASP_Z_ABOVE = 0.10   # hover/lift/carry height
    GRASP_Z = 0.02         # descend-to-grasp EE height
    # Extra clearance above the computed place height so the cube is released
    # just above the base top instead of being driven into it.
    STACK_PLACE_CLEARANCE = 0.002

    # Workspace used when all four colored cylinders are visible at once.
    # Compared with the previous x=(-0.18, 0.18), y=(0.10, 0.18), this keeps
    # objects slightly farther forward and more centered left-to-right.
    DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
    DEFAULT_OBJECT_Y_RANGE = (0.16, 0.20)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.035

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False,
                 noslip_iterations=None, camera_jitter=0.0):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # noslip_iterations 가 지정되면 XML 의 <option noslip_iterations> 를 런타임에 덮어쓴다.
        # 마찰(접선) 잔류 slip 을 줄이는 후처리 PGS 패스 횟수로, 높을수록 그래스핑이 단단하지만
        # 스텝당 물리 비용이 커진다. None 이면 XML 값을 그대로 둔다.
        if noslip_iterations is not None:
            self.model.opt.noslip_iterations = int(noslip_iterations)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer

        # Per-episode camera-position jitter (domain randomization). 0.0 = off, so
        # the grasp (010) path that never sets this keeps its fixed XML viewpoint.
        # Resolve the named camera's id once and stash its nominal XML pose so jitter
        # always perturbs around that anchor instead of drifting across episodes.
        self.camera_jitter = float(camera_jitter)
        self._cam_id = -1
        if camera_name is not None:
            self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if self._cam_id == -1:
                raise ValueError(f"camera not found: {camera_name!r}")
            self._cam_pos0 = self.model.cam_pos[self._cam_id].copy()

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = self.CYLINDER_BODY_BY_COLOR["red"]

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        # Initialize all colored cylinders in the scene. Dataset collection will
        # randomize these positions for every episode.
        self.reset_episode(
            object_specs=self.make_default_object_specs(),
            target_color="red",
        )

    # ---------- kinematics / commands ----------

    def _calc_inv_kinematics(self, x, y, z):
        """
        Inputs are in centimeters, matching the uploaded move_to code style.
        Returns [j1, j2, j3, j4] in degrees.
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
            if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
                x, y = y, -x
                th1 = math.atan2(y, x)
                c1 = math.cos(th1)
                s1 = math.sin(th1)
                x = x - self.L4 * c1
                y = y - self.L4 * s1
                zL1 = z - self.L1
                c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
                c32 = c3 * c3
                if c32 > 1:
                    c32 = 1
                s3 = -math.sqrt(1 - c32)
                th3 = math.atan2(s3, c3)
                M1 = c3 * self.L3 + self.L2
                M2 = z - self.L1
                M3 = s3 * self.L3
                M4 = c1 * x + s1 * y
                c2 = M1 * M2 - M3 * M4
                s2 = -M2 * M3 - M1 * M4
                th2 = math.atan2(s2, c2)
                th1 = math.degrees(th1)
                th2 = math.degrees(th2)
                th3 = math.degrees(th3)
                th4 = -(th2 + th3) - 90

                if th1 < -120 or th1 > 120:
                    return None
                if th2 < -90 or th2 > 30:
                    return None
                if th3 < -150 or th3 > 0:
                    return None

                return [th1, th2, th3, th4]
            return None
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]

        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = np.clip(speed, 0.0, 100.0)
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self):
        self.gripper_target = self.GRIP_CLOSE

    def lockh(self):
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        """
        action = [target_x_m, target_y_m, target_z_m, gripper]
        """
        target_x, target_y, target_z, gripper = action

        # move_to convention is centimeters.
        self.move_to(target_x * 100.0, target_y * 100.0, target_z * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    # ---------- synchronous stepping ----------

    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                else:
                    desired = base_angle - np.radians(180)

                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        # Gripper stop-on-contact logic from uploaded code.
        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------

    def get_robot_state(self):
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {
            "joint_angles": joint_angles,
            "gripper_state": gripper_state
        }

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])

        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def get_body_xy(self, body_name):
        """World xy of a body origin (used for stack src/base placement)."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        pos = self.data.xpos[body_id]
        return float(pos[0]), float(pos[1])

    def get_body_top_z(self, body_name):
        """Top-surface z of a body's first geom = geom world z + z half-extent.

        XML-driven: reads geom_xpos/geom_size from the compiled model rather than
        hardcoding the base height, so changing the base size in the XML keeps the
        place height correct without touching this code.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        geom_num = int(self.model.body_geomnum[body_id])
        if geom_num < 1:
            raise ValueError(f"{body_name} has no geom")
        geom_id = int(self.model.body_geomadr[body_id])
        return float(self.data.geom_xpos[geom_id][2] + self.model.geom_size[geom_id][2])

    def get_body_linear_speed(self, body_name):
        """Linear speed (m/s) of a free-joint body. 0.0 if the body has no joint
        (e.g. the static base). Used to confirm a placed cube has settled."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        if int(self.model.body_jntnum[body_id]) < 1:
            return 0.0
        joint_id = int(self.model.body_jntadr[body_id])
        qvel_adr = int(self.model.jnt_dofadr[joint_id])
        return float(np.linalg.norm(self.data.qvel[qvel_adr:qvel_adr + 3]))

    def randomize_camera(self, rng):
        """Per-episode camera-position jitter for viewpoint domain randomization.

        Shifts the named camera by a small uniform xyz offset (±camera_jitter
        meters per axis) around its nominal XML pose, so each demo is rendered
        from a slightly different viewpoint. Orientation is left untouched. The
        offset is applied relative to the stored anchor pose (_cam_pos0), so it
        never accumulates across episodes. No-op when camera_jitter<=0 or there
        is no named camera (grasp/010 path keeps its fixed viewpoint).
        """
        if self.camera_jitter <= 0.0 or self._cam_id == -1:
            return
        offset = rng.uniform(-self.camera_jitter, self.camera_jitter, size=3)
        self.model.cam_pos[self._cam_id] = self._cam_pos0 + offset

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_ee_pose(self, body_name="Link4"):
        """Gripper-tip EE pose in meters via forward kinematics.

        Must stay identical to raccoon_env.SyncSimRaccoonEnv.get_ee_pose and be the
        inverse of _calc_inv_kinematics()/move_to(), which command the *tip* (Link4
        origin + L4). Do NOT log the MuJoCo body xpos of Link4 as ee_pose: that origin
        sits ~L4 (8cm) short of the controlled tip and the offset rotates with joint 1,
        so xpos deltas describe a different point than execute_delta_action7 moves at
        rollout time — which makes the closed-loop policy miss every grasp. Using this
        same FK for the recorded ee_pose keeps the action-label frame and the
        execution frame identical (train == rollout).
        """
        th1 = float(self.data.qpos[0])
        th2 = float(self.data.qpos[1])
        th3 = float(self.data.qpos[2])

        r = -self.L2 * math.sin(th2) - self.L3 * math.sin(th2 + th3)
        z = self.L1 + self.L2 * math.cos(th2) + self.L3 * math.cos(th2 + th3)
        r_tip = r + self.L4

        x_cm = -math.sin(th1) * r_tip
        y_cm = math.cos(th1) * r_tip
        z_cm = z
        return x_cm / 100.0, y_cm / 100.0, z_cm / 100.0

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name

        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()

        # EE pose: forward-kinematics gripper tip (the point move_to()/IK and the
        # rollout's execute_delta_action7 actually control). Logging this — not the
        # Link4 body xpos — keeps the recorded action deltas in the same frame the
        # policy executes at rollout time. See get_ee_pose().
        ee_pose_list = list(self.get_ee_pose())

        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    # ---------- reset / success ----------

    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        # Zero object joint velocities if present.
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    @classmethod
    def make_default_object_specs(cls):
        """
        Deterministic fallback placement for initialization only.
        Dataset collection uses sample_object_specs() for randomized positions.
        """
        x_values = np.linspace(
            cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75,
            cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75,
            len(cls.CYLINDER_COLORS),
        )
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {
            color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": float(x_values[idx]),
                "y": y_center,
                "yaw": 0.0,
            }
            for idx, color in enumerate(cls.CYLINDER_COLORS)
        }

    @classmethod
    def sample_object_specs(
        cls,
        rng,
        colors=None,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
        min_distance=None,
        max_tries=1000,
        reserved_xy=None,
    ):
        """
        Randomly place all colored cylinders in the visible workspace.

        Defaults intentionally narrow the spawn area compared with the older
        single-object collector:
          - x: -0.18~0.18  ->  -0.10~0.10
          - y:  0.10~0.18  ->   0.16~0.20
        A minimum XY distance prevents blocks from overlapping or touching.

        reserved_xy (optional [x, y]) pre-seeds the occupancy list so no object is
        placed within min_distance of it. The stack task passes the static base xy
        here so colored cubes never spawn on top of the base.
        """
        colors = tuple(colors or cls.CYLINDER_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance

        if len(colors) == 0:
            raise ValueError("colors는 비어 있을 수 없습니다.")
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

        specs = {}
        placed_xy = []
        if reserved_xy is not None:
            placed_xy.append(np.array(reserved_xy, dtype=np.float64))
        # Shuffle placement order so one color is not systematically favored.
        placement_order = list(colors)
        rng.shuffle(placement_order)

        for color in placement_order:
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")

            for _ in range(max_tries):
                x = float(rng.uniform(x_range[0], x_range[1]))
                y = float(rng.uniform(y_range[0], y_range[1]))
                xy = np.array([x, y], dtype=np.float64)

                if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                    specs[color] = {
                        "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                        "x": x,
                        "y": y,
                        "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                    }
                    placed_xy.append(xy)
                    break
            else:
                raise RuntimeError(
                    "색상 cylinder 4개를 겹치지 않게 배치하지 못했습니다. "
                    f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
                )

        # Return in canonical color order for stable metadata.
        return {color: specs[color] for color in colors}

    @staticmethod
    def specs_to_meta(object_specs):
        return {
            color: {
                "body_name": str(spec["body_name"]),
                "xy": [float(spec["x"]), float(spec["y"])],
                "yaw": float(spec["yaw"]),
            }
            for color, spec in object_specs.items()
        }

    def reset_colored_objects(self, object_specs, target_color):
        """
        Place every colored cylinder in the scene. The target color controls
        which body is used for object_pose logging and grasp trajectory target.
        """
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

        self.active_object_body_name = object_specs[target_color]["body_name"]

        for color, spec in object_specs.items():
            body_name = spec["body_name"]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"body not found for color '{color}': {body_name}")

            self.reset_object_pose(
                body_name,
                x=spec["x"],
                y=spec["y"],
                z=0.02,
                yaw=spec["yaw"],
            )

    def reset_episode(self, object_specs, target_color="red"):
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)
        mujoco.mj_forward(self.model, self.data)

        # Short stabilization after reset.
        self.step_n(20)

    def get_gripper_touch_state(self):
        """
        Return whether the left/right gripper touch sensors are in contact.
        If the XML does not expose these sensors, this returns False for both sides.
        """
        try:
            touch_l = float(self.data.sensor("sensor_L").data[0])
            touch_r = float(self.data.sensor("sensor_R").data[0])
        except Exception:
            touch_l = 0.0
            touch_r = 0.0

        return touch_l, touch_r

    def is_grasp_success(self, touch_threshold=0.1, require_closed=True):
        """
        Grasp-only success criterion.
        The episode is considered successful when both gripper touch sensors detect contact.
        Optionally also require the gripper to have moved away from its fully-open position.
        """
        touch_l, touch_r = self.get_gripper_touch_state()
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)

        if not require_closed:
            return bool(both_touched)

        # Make sure this is not just an accidental touch while the gripper is still fully open.
        gripper_is_closing_or_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        return bool(both_touched and gripper_is_closing_or_closed)

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        """
        Return True when the requested object body is in contact with a non-floor,
        non-cylinder body. This makes success target-specific when all four
        colored cylinders are present: touching the wrong color does not count.
        """
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        cylinder_body_ids = set()
        for cylinder_body_name in self.CYLINDER_BODY_BY_COLOR.values():
            cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_body_name)
            if cylinder_body_id != -1:
                cylinder_body_ids.add(cylinder_body_id)

        ignored_geom_names = set(ignored_geom_names or [])

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if target_body_id not in (body1, body2):
                continue

            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1

            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue

            # Do not count target-object contact with another colored cylinder
            # as a grasp. We only want contacts against the robot/gripper.
            if other_body in cylinder_body_ids:
                continue

            return True

        return False

    def is_bodies_touching(self, body_a_name, body_b_name):
        """True when bodies a and b share at least one active contact pair.

        Same contact walk as is_body_touching_robot, but the filter is inverted to
        the (a, b) pair — used to confirm a stacked cube is resting on the base.
        """
        a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_a_name)
        b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_b_name)
        if a_id == -1:
            raise ValueError(f"body not found: {body_a_name}")
        if b_id == -1:
            raise ValueError(f"body not found: {body_b_name}")

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if {body1, body2} == {a_id, b_id}:
                return True
        return False

    def is_target_grasp_success(self, target_body_name, touch_threshold=0.1, require_closed=True):
        """
        Success for the multi-cylinder scene. Both gripper touch sensors must be
        active, the gripper must be closing/closed, and the prompted target body
        must be the object contacting the robot.
        """
        return bool(
            self.is_grasp_success(touch_threshold=touch_threshold, require_closed=require_closed)
            and self.is_body_touching_robot(target_body_name)
        )

    def is_stack_success(
        self,
        src_body_name,
        base_body_name="stack_base",
        xy_tol=0.013,
        z_tol=0.006,
        speed_tol=0.05,
        touch_threshold=0.1,
    ):
        """Pick-and-stack success: the source cube is resting, released, centered
        on the base.

        Unlike grasp success (which is a touch-while-closing check), stacking is
        about the cube settling on the base *after* release, so the criteria are:
          1. src xy within xy_tol of base xy (centered)
          2. src z ≈ base_top_z + CUBE_HALF_HEIGHT (sitting one cube-height up)
          3. src and base share a contact (actually resting on the base)
          4. gripper open and touch sensors low (the cube was released)
          5. src nearly motionless (settled, not mid-bounce)
        """
        src_xy = np.array(self.get_body_xy(src_body_name), dtype=np.float64)
        base_xy = np.array(self.get_body_xy(base_body_name), dtype=np.float64)
        if float(np.linalg.norm(src_xy - base_xy)) > xy_tol:
            return False

        base_top_z = self.get_body_top_z(base_body_name)
        src_z = float(self.get_object_pose(src_body_name)[2])
        if abs(src_z - (base_top_z + self.CUBE_HALF_HEIGHT)) > z_tol:
            return False

        if not self.is_bodies_touching(src_body_name, base_body_name):
            return False

        # Released: gripper not clamped on the cube anymore.
        touch_l, touch_r = self.get_gripper_touch_state()
        if (touch_l > touch_threshold) and (touch_r > touch_threshold):
            return False
        gripper_open = float(self.data.qpos[4]) > (self.GRIP_OPEN - 0.02)
        if not gripper_open:
            return False

        # Settled.
        if self.get_body_linear_speed(src_body_name) > speed_tol:
            return False

        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---------- grasp-only plan ----------

    def make_grasp_plan(self, box_x, box_y):
        z_above = self.GRASP_Z_ABOVE
        z_grasp = self.GRASP_Z

        return [
            [box_x, box_y, z_above, 0],   # Move above object with gripper open.
            [box_x, box_y, z_grasp, 0],   # Move down to grasp height.
            [box_x, box_y, z_grasp, 1],   # Close gripper and finish once the object is grasped.
        ]

    # ---------- pick-and-stack plan ----------

    def make_stack_plan(self, src_x, src_y, base_x, base_y, base_top_z):
        """8-waypoint pick-and-stack trajectory (EE targets [x, y, z, grip]).

        Reuses the grasp hover/grasp heights, then adds lift -> carry -> place ->
        release -> retract. The converter turns consecutive ee_pose deltas + the
        gripper bit into 7D actions automatically, so only the waypoints differ.

        Place height (EE z) reuses the grasp EE offset: when grasping a cube
        resting on the floor at center z=CUBE_HALF_HEIGHT, the EE sits at GRASP_Z,
        so the EE is GRASP_Z - CUBE_HALF_HEIGHT above the held cube center. To land
        the cube center at base_top_z + CUBE_HALF_HEIGHT, the EE must be that same
        offset (plus a little clearance) above the target cube center.
        """
        z_above = self.GRASP_Z_ABOVE
        z_grasp = self.GRASP_Z
        ee_offset = self.GRASP_Z - self.CUBE_HALF_HEIGHT
        z_place = (
            base_top_z + self.CUBE_HALF_HEIGHT + ee_offset + self.STACK_PLACE_CLEARANCE
        )

        return [
            [src_x, src_y, z_above, 0],    # above src, open
            [src_x, src_y, z_grasp, 0],    # descend, open
            [src_x, src_y, z_grasp, 1],    # close on src
            [src_x, src_y, z_above, 1],    # lift (hold)
            [base_x, base_y, z_above, 1],  # carry over base (hold)
            [base_x, base_y, z_place, 1],  # descend onto base top (hold)
            [base_x, base_y, z_place, 0],  # release (open)
            [base_x, base_y, z_above, 0],  # retract up
        ]


def run_episode_and_record(
    rc: SyncSimRaccoonDataset,
    logger: DatasetLogger,
    episode_id: int,
    instruction: str,
    object_specs: dict,
    target_color: str = "red",
    task: str = "grasp",
    base_body_name: str = "stack_base",
    speed: int = 70,
    settle_seconds_per_action: float = 2.0,
    initial_settle_seconds: float = 0.3,
    hz: int = 10,
    touch_threshold: float = 0.1,
):
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
    if task not in ("grasp", "stack"):
        raise ValueError(f"지원하지 않는 task: {task} (grasp|stack)")

    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()

    # Let newly reset free-joint cylinders fall/settle before capturing frame_000000.
    # Without this, the first saved image can show cylinders slightly floating while
    # later frames look normal after one physics step.
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)

    # Build the task plan, success criterion, and metadata. Both branches share
    # the exact same logging loop below — only the waypoints / success_fn differ.
    if task == "grasp":
        task_type = "grasp"
        goal_xy = [target_x, target_y]
        box_init_xy = [target_x, target_y]
        extra_meta = None
        # The prompt decides which cylinder to grasp. All four are visible, but
        # the trajectory is aimed only at the prompted color.
        plan = rc.make_grasp_plan(target_x, target_y)

        def success_fn():
            return rc.is_target_grasp_success(
                target_body_name=target_body_name,
                touch_threshold=touch_threshold,
            )
    else:  # task == "stack"
        task_type = "pick_and_stack"
        # Use the settled cube/base world xy (more accurate than spawn targets).
        src_x, src_y = rc.get_body_xy(target_body_name)
        base_x, base_y = rc.get_body_xy(base_body_name)
        base_top_z = rc.get_body_top_z(base_body_name)
        # episode_metadata schema is fixed: reuse goal_xy=base xy (place target)
        # and box_init_xy=src xy (pick source). Stack-only extras go in extra_meta.
        goal_xy = [base_x, base_y]
        box_init_xy = [src_x, src_y]
        extra_meta = {
            "base_body_name": str(base_body_name),
            "base_xy": [base_x, base_y],
            "base_top_z": float(base_top_z),
        }
        plan = rc.make_stack_plan(src_x, src_y, base_x, base_y, base_top_z)

        def success_fn():
            return rc.is_stack_success(
                src_body_name=target_body_name,
                base_body_name=base_body_name,
                touch_threshold=touch_threshold,
            )

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type=task_type,
        goal_xy=goal_xy,
        box_init_xy=box_init_xy,
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
        extra_meta=extra_meta,
    )

    try:
        # Initial observation.
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0

        for action in plan:
            # Set control target to current waypoint.
            rc.execute_action(action, speed=speed)

            # Capture continuous observations at specified Hz while moving toward the target.
            num_frames = int(settle_seconds_per_action * hz)

            for _ in range(num_frames):
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )

                # Advance physics by dt seconds.
                rc.settle_steps(seconds=dt)

                # Observe after stepping.
                obs = rc.get_observation()
                step_counter += 1

        # Record terminal observation.
        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        success = success_fn()
        logger.finalize_episode(success=success)
        return success

    except Exception as e:
        logger.abort_episode()
        raise e


def _balanced_target_counts(num_episodes, colors):
    """
    Return per-color episode targets. If num_episodes is divisible by the
    number of colors, the split is exactly equal. Otherwise the remainder is
    distributed one-by-one to the first colors.
    """
    base = num_episodes // len(colors)
    remainder = num_episodes % len(colors)
    return {
        color: base + (1 if idx < remainder else 0)
        for idx, color in enumerate(colors)
    }


def _sample_remaining_color(rng, target_counts, success_counts):
    remaining_colors = []
    remaining_weights = []

    for color, target_count in target_counts.items():
        remaining = target_count - success_counts[color]
        if remaining > 0:
            remaining_colors.append(color)
            remaining_weights.append(remaining)

    if not remaining_colors:
        return None

    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    return str(rng.choice(remaining_colors, p=remaining_weights))


def collect_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_grasp_colored_cylinder",
    num_episodes=100,
    colors=("red", "blue", "green", "yellow"),
    instruction_template="grasp the {color} cylinder",
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    camera_jitter=0.0,
    speed=150,
    settle_seconds_per_action=0.8,
    initial_settle_seconds=0.3,
    hz=10,
    touch_threshold=0.1,
    noslip_iterations=None,
    seed=None,
    max_attempts=None,
    object_x_range=(-0.10, 0.10),
    object_y_range=(0.16, 0.20),
    min_object_distance=0.035,
    episode_id_offset=0,
    worker_label=None,
    task="grasp",
    base_body_name="stack_base",
    base_label="base",
    progress_success=None,
    progress_attempts=None,
):
    """
    Collect a balanced grasp dataset for colored cylinders.

    Each episode contains all four colored cylinders at randomized positions.
    The instruction selects which colored cylinder is the target, and the robot
    executes the grasp plan toward that target color only.

    Default behavior with keep_failed=False:
    - Saves exactly num_episodes successful episodes when possible.
    - Balances successful episodes across colors according to target_counts.
      For num_episodes=500 and 4 colors, this yields 125 episodes per color.
    - Failed episodes are discarded and retried with the remaining color quota.
    - Before frame_000000 is captured, the scene is stepped for
      initial_settle_seconds so free-joint cylinders are already resting on the table.

    Position defaults are constrained relative to the old single-object range:
    - old x range: -0.18~0.18  ->  new x range: -0.10~0.10
    - old y range:  0.10~0.18  ->  new y range:  0.16~0.20

    If keep_failed=True, failed episodes are also saved, so the final folder can
    contain more than num_episodes attempts and the all-attempt ratio may differ.
    """
    colors = tuple(colors)
    valid_colors = set(SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.keys())
    unknown_colors = [color for color in colors if color not in valid_colors]
    if unknown_colors:
        raise ValueError(f"지원하지 않는 색상입니다: {unknown_colors}. 지원 색상: {sorted(valid_colors)}")

    if len(colors) == 0:
        raise ValueError("colors는 비어 있을 수 없습니다.")
    if task not in ("grasp", "stack"):
        raise ValueError(f"지원하지 않는 task: {task} (grasp|stack)")

    target_counts = _balanced_target_counts(num_episodes, colors)
    rng = np.random.default_rng(seed)

    if max_attempts is None:
        # Prevent infinite loops if the scripted expert repeatedly fails. Stacking
        # is a longer/harder trajectory than grasp, so give it a larger budget.
        per_episode = 40 if task == "stack" else 20
        max_attempts = max(num_episodes * per_episode, num_episodes + 100)

    rc = SyncSimRaccoonDataset(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
        noslip_iterations=noslip_iterations,
        camera_jitter=camera_jitter,
    )
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)

    # For stacking, the static base xy is fixed for the whole run. Reserve it so
    # no colored cube ever spawns on top of the base.
    reserved_xy = None
    if task == "stack":
        reserved_xy = rc.get_body_xy(base_body_name)

    success_counts = {color: 0 for color in colors}
    attempt_count = 0

    label = f"[{worker_label}] " if worker_label is not None else ""
    print(f"{label}Target color counts: {target_counts} (episode_id_offset={episode_id_offset})")

    t_start = time.monotonic()
    try:
        while sum(success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            target_color = _sample_remaining_color(rng, target_counts, success_counts)
            if target_color is None:
                attempt_count -= 1  # 이번 반복은 실제 시도가 아니므로 카운트 되돌림
                break

            did_succeed = False

            # base= is ignored by templates without a {base} field (e.g. grasp),
            # so passing it is forward-compatible with a colored-base instruction.
            instruction = instruction_template.format(color=target_color, base=base_label)
            object_specs = SyncSimRaccoonDataset.sample_object_specs(
                rng=rng,
                colors=colors,
                x_range=object_x_range,
                y_range=object_y_range,
                min_distance=min_object_distance,
                reserved_xy=reserved_xy,
            )

            # Slightly vary the camera viewpoint for every demo (no-op if
            # camera_jitter<=0). Uses the same per-worker rng so the viewpoint
            # stream is reproducible from --seed.
            rc.randomize_camera(rng)

            # With keep_failed=False, failed attempts are deleted, so reusing the
            # next successful episode id keeps folder numbering compact.
            #
            # episode_id_offset가 worker별로 분리되어 있어서, ProcessPoolExecutor로
            # 여러 프로세스가 같은 dataset_root에 써도 episode id가 절대 겹치지 않는다.
            local_id = attempt_count if keep_failed else (sum(success_counts.values()) + 1)
            episode_id = episode_id_offset + local_id

            try:
                success = run_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    instruction=instruction,
                    object_specs=object_specs,
                    target_color=target_color,
                    task=task,
                    base_body_name=base_body_name,
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    touch_threshold=touch_threshold,
                )

                if success:
                    success_counts[target_color] += 1
                    did_succeed = True

                print(
                    f"{label}[Attempt {attempt_count:04d}] episode_id={episode_id:06d} | "
                    f"task='{task}' | color='{target_color}' | "
                    f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
                    f"instruction='{instruction}' | success={success} | "
                    f"success_counts={success_counts}"
                )
            except Exception as e:
                print(
                    f"{label}[Attempt {attempt_count:04d}] task='{task}' | "
                    f"color='{target_color}' | exception: {e}"
                )

            # 부모의 진행률 모니터가 읽는 공유메모리 카운터 갱신(있을 때만).
            # 각 Value 는 자체 lock(세마포어)을 가지며, 증가 연산 1회만 직렬화한다.
            # 무거운 작업(sim/render/save)은 전부 이 블록 밖이라 경합이 거의 없다.
            if progress_attempts is not None:
                with progress_attempts.get_lock():
                    progress_attempts.value += 1
            if did_succeed and progress_success is not None:
                with progress_success.get_lock():
                    progress_success.value += 1

    finally:
        rc.close()

    elapsed = max(1e-9, time.monotonic() - t_start)
    total_success = sum(success_counts.values())
    succ_per_sec = total_success / elapsed
    att_per_sec = attempt_count / elapsed
    print(f"{label}완료: success episodes = {total_success}/{num_episodes}, attempts = {attempt_count}")
    print(f"{label}색상별 성공 episode 수: {success_counts}")
    print(f"{label}throughput: {succ_per_sec:.3f} success-episodes/sec, "
          f"{att_per_sec:.3f} attempts/sec (elapsed {elapsed:.1f}s)")

    if total_success < num_episodes:
        print(
            f"{label}주의: max_attempts에 도달해서 목표 episode 수를 모두 채우지 못했습니다. "
            "max_attempts를 늘리거나 grasp 성공 조건/동작 파라미터를 확인하세요."
        )

    return {
        "success_counts": success_counts,
        "total_success": total_success,
        "attempt_count": attempt_count,
        "num_episodes": num_episodes,
    }


# 진행률 공유메모리 카운터. Value 는 submit 인자(런타임 큐)로는 전달할 수 없고
# 프로세스 생성 시 상속으로만 전달되므로, 부모가 ProcessPoolExecutor 의 initializer 로
# 각 worker 전역에 한 번 바인딩한 뒤 _collect_worker 에서 collect_dataset 로 넘긴다.
_PROGRESS_SUCCESS = None
_PROGRESS_ATTEMPTS = None


def _init_progress_counters(success_counter, attempts_counter):
    """ProcessPoolExecutor initializer: 공유 카운터를 worker 전역에 바인딩한다."""
    global _PROGRESS_SUCCESS, _PROGRESS_ATTEMPTS
    _PROGRESS_SUCCESS = success_counter
    _PROGRESS_ATTEMPTS = attempts_counter


def _collect_worker(worker_kwargs):
    """
    ProcessPoolExecutor에 넘길 모듈 레벨 entry point.

    각 프로세스는 이 함수 안에서 자신만의 MjModel / MjData / Renderer(EGL 컨텍스트)를
    새로 만든다(collect_dataset 내부에서 SyncSimRaccoonDataset를 생성). MuJoCo 객체는
    pickle이 불가능하므로 부모에서 만들어 넘기는 대신 worker별로 생성해야 한다.
    """
    return collect_dataset(
        progress_success=_PROGRESS_SUCCESS,
        progress_attempts=_PROGRESS_ATTEMPTS,
        **worker_kwargs,
    )


def _format_hms(seconds):
    """초를 사람이 읽기 쉬운 'Hh Mm Ss' 문자열로."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _run_progress_monitor(success_counter, attempts_counter, total, t_start,
                          stop_event, interval=5.0):
    """부모 프로세스에서 주기적으로 집계 진행률을 출력한다.

    worker들이 공유메모리 Value(success/attempts)에 누적한 값을 읽어
    'n/N개, 경과 시간, 예상 남은 시간'을 라이브로 보여 준다. 데이터 생성이
    끝나면 stop_event 가 set 되어 루프가 종료된다. 표시용 읽기는 lock 없이
    수행해(워커 증가 경로와 경합하지 않음) 카운터 오버헤드를 0에 가깝게 둔다.
    """
    while not stop_event.wait(interval):
        done = int(success_counter.value)
        attempts = int(attempts_counter.value)
        elapsed = time.monotonic() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * done / total if total else 0.0
        if done > 0:
            eta = elapsed * (total - done) / done
            eta_str = f"~{_format_hms(eta)}"
        else:
            eta_str = "예상 계산 중"
        print(
            f"[진행] 성공 {done}/{total} ({pct:.1f}%) | 시도 {attempts}회 | "
            f"경과 {_format_hms(elapsed)} | 남은 예상 {eta_str} | {rate:.2f} ep/s",
            flush=True,
        )


def collect_dataset_parallel(
    num_workers=12,
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_grasp_colored_cylinder",
    num_episodes=400,
    colors=("red", "blue", "green", "yellow"),
    instruction_template="grasp the {color} cylinder",
    keep_failed=False,
    camera_name="front_view",
    camera_jitter=0.0,
    speed=150,
    settle_seconds_per_action=0.8,
    initial_settle_seconds=0.1,
    hz=10,
    touch_threshold=0.1,
    noslip_iterations=None,
    seed=None,
    max_attempts_per_worker=None,
    object_x_range=(-0.10, 0.10),
    object_y_range=(0.16, 0.25),
    min_object_distance=0.035,
    task="grasp",
    base_body_name="stack_base",
    base_label="base",
    episode_id_offset_base=0,
):
    """
    ProcessPoolExecutor로 MuJoCo 데이터 수집을 멀티코어 병렬 실행한다.

    설계 포인트:
    - num_episodes를 num_workers개로 균등 분할(나머지는 앞쪽 worker부터 +1)하여
      각 worker가 자신의 성공 episode 목표량을 독립적으로 채운다.
    - 각 worker는 서로 겹치지 않는 episode_id_offset을 받는다. offset은
      "그 앞 worker들의 max_attempts 합"으로 누적 계산하므로, keep_failed 여부와
      무관하게 모든 worker의 episode id 구간이 서로소(disjoint)가 된다.
        worker w의 id 범위 = [offset_w + 1, offset_w + max_attempts_w]
    - 각 worker는 seed+w로 서로 다른 RNG를 사용해 동일 장면이 중복 생성되는 것을 막는다.
    - viewer는 병렬 환경에서 의미가 없으므로 항상 False.
    - GL 컨텍스트가 fork로 복제되어 깨지는 것을 막기 위해 spawn 컨텍스트를 쓴다.
    """
    import multiprocessing as mp

    if num_workers < 1:
        raise ValueError("num_workers는 1 이상이어야 합니다.")
    num_workers = min(num_workers, num_episodes)

    # num_episodes를 worker별로 균등 분할.
    base = num_episodes // num_workers
    remainder = num_episodes % num_workers
    per_worker_episodes = [base + (1 if w < remainder else 0) for w in range(num_workers)]

    # worker별 max_attempts (id 구간 stride 계산에 사용). 스태킹은 더 어려운
    # 궤적이라 episode당 시도 예산을 늘린다(collect_dataset 기본과 일치).
    _per_episode_budget = 40 if task == "stack" else 20

    def _default_max_attempts(n):
        return max(n * _per_episode_budget, n + 100)

    per_worker_max_attempts = [
        max_attempts_per_worker if max_attempts_per_worker is not None
        else _default_max_attempts(n)
        for n in per_worker_episodes
    ]

    # worker별 id 구간 stride.
    # - keep_failed=False: 성공한 episode에만 id를 붙이고 local_id는 1..n_w 범위이므로
    #   stride를 n_w로 잡으면 전체 id가 1..num_episodes로 촘촘하게 채워진다.
    # - keep_failed=True: 실패 attempt에도 id(=attempt_count)를 쓰므로 local_id가
    #   1..max_attempts_w까지 갈 수 있어 stride로 max_attempts를 써야 충돌이 없다.
    per_worker_stride = (
        list(per_worker_max_attempts) if keep_failed else list(per_worker_episodes)
    )

    # 누적 offset -> id 구간 disjoint 보장. episode_id_offset_base 로 전체 구간을
    # 평행이동할 수 있다(예: stack 은 1_000_000 부터 시작해 grasp episode id 와 분리).
    offsets = []
    acc = episode_id_offset_base
    for stride in per_worker_stride:
        offsets.append(acc)
        acc += stride

    Path(dataset_root).mkdir(parents=True, exist_ok=True)

    jobs = []
    for w in range(num_workers):
        worker_seed = None if seed is None else seed + w
        jobs.append({
            "xml_path": xml_path,
            "dataset_root": dataset_root,
            "num_episodes": per_worker_episodes[w],
            "colors": colors,
            "instruction_template": instruction_template,
            "keep_failed": keep_failed,
            "use_viewer": False,
            "camera_name": camera_name,
            "camera_jitter": camera_jitter,
            "speed": speed,
            "settle_seconds_per_action": settle_seconds_per_action,
            "initial_settle_seconds": initial_settle_seconds,
            "hz": hz,
            "touch_threshold": touch_threshold,
            "noslip_iterations": noslip_iterations,
            "seed": worker_seed,
            "max_attempts": per_worker_max_attempts[w],
            "object_x_range": object_x_range,
            "object_y_range": object_y_range,
            "min_object_distance": min_object_distance,
            "episode_id_offset": offsets[w],
            "worker_label": f"W{w:02d}",
            "task": task,
            "base_body_name": base_body_name,
            "base_label": base_label,
        })

    print(
        f"병렬 수집 시작: num_workers={num_workers}, num_episodes={num_episodes}, "
        f"worker별 episode={per_worker_episodes}, id offsets={offsets}"
    )

    total_success = 0
    total_attempts = 0
    aggregated_color_counts = {color: 0 for color in colors}

    # worker들이 success/attempts 를 누적할 공유메모리 원자 카운터. Manager 같은 별도
    # 프로세스/소켓 IPC 없이 공유메모리 + 세마포어로만 동작하므로, 128코어에서 모든
    # worker가 동시에 갱신해도 직렬화 병목이 없다(증가 1회만 락). spawn 컨텍스트에서
    # Value 는 submit 인자로는 못 넘기고 프로세스 생성 시 상속으로만 전달되므로,
    # ProcessPoolExecutor 의 initializer/initargs 로 각 worker 전역에 주입한다.
    ctx = mp.get_context("spawn")
    success_counter = ctx.Value("L", 0)
    attempts_counter = ctx.Value("L", 0)

    t_start = time.monotonic()

    # 생성 도중 진행률(n/N, 경과/예상 시간)을 주기적으로 출력하는 모니터 스레드.
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_run_progress_monitor,
        args=(success_counter, attempts_counter, num_episodes, t_start, stop_event),
        daemon=True,
    )
    monitor.start()

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_progress_counters,
            initargs=(success_counter, attempts_counter),
        ) as executor:
            future_to_worker = {
                executor.submit(_collect_worker, job): job["worker_label"]
                for job in jobs
            }
            for future in as_completed(future_to_worker):
                label = future_to_worker[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[{label}] worker 실패: {e}")
                    continue

                total_success += result["total_success"]
                total_attempts += result["attempt_count"]
                for color, count in result["success_counts"].items():
                    aggregated_color_counts[color] = aggregated_color_counts.get(color, 0) + count
    finally:
        stop_event.set()
        monitor.join(timeout=2.0)

    elapsed = max(1e-9, time.monotonic() - t_start)
    succ_per_sec = total_success / elapsed
    att_per_sec = total_attempts / elapsed
    print(
        f"전체 완료: success episodes = {total_success}/{num_episodes}, "
        f"attempts = {total_attempts}"
    )
    print(f"색상별 성공 episode 수(합산): {aggregated_color_counts}")
    print(
        f"throughput(전체): {succ_per_sec:.3f} success-episodes/sec, "
        f"{att_per_sec:.3f} attempts/sec | num_workers={num_workers}, "
        f"elapsed {elapsed:.1f}s"
    )

    return {
        "total_success": total_success,
        "total_attempts": total_attempts,
        "success_counts": aggregated_color_counts,
        "elapsed_s": elapsed,
        "success_per_sec": succ_per_sec,
        "attempts_per_sec": att_per_sec,
        "num_workers": num_workers,
    }


# task 별 기본값(xml / dataset-root / instruction / episode-id offset base).
# stack 의 offset base 를 1_000_000 으로 두어, 같은 intermediate out_root 로 변환할 때
# grasp(episode_0000xx)와 stack(episode_10000xx)의 id 구간이 절대 겹치지 않게 한다.
TASK_DEFAULTS = {
    "grasp": {
        "xml_path": "Raccoon_colored_cylinder.xml",
        "dataset_root": "raccoon_grasp_colored_cylinder",
        "instruction_template": "grasp the {color} cylinder",
        "episode_id_offset_base": 0,
    },
    "stack": {
        "xml_path": "Raccoon_stack_scene.xml",
        "dataset_root": "raccoon_pick_and_stack",
        "instruction_template": "stack the {color} cube on the {base}",
        "episode_id_offset_base": 1_000_000,
    },
}


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="MuJoCo colored-object grasp/stack 데이터셋을 멀티코어로 병렬 수집한다."
    )
    parser.add_argument(
        "--task", type=str, default="grasp", choices=("grasp", "stack"),
        help="수집할 task. grasp(기본, 하위호환) 또는 stack(pick-and-stack).",
    )
    parser.add_argument(
        "--num-workers", "-j", type=str, default="12",
        help="병렬 실행할 프로세스(코어) 수. 정수 또는 'auto'(코어 수 기반 자동) (기본: 12)",
    )
    parser.add_argument(
        "--num-episodes", "-n", type=int, default=400,
        help="수집할 성공 episode 총 개수 (기본: 400)",
    )
    parser.add_argument(
        "--xml-path", type=str, default=None,
        help="MuJoCo XML 경로 (미지정 시 task 기본값)",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="데이터셋 출력 루트 디렉토리 (미지정 시 task 기본값)",
    )
    parser.add_argument(
        "--instruction-template", type=str, default=None,
        help="instruction 템플릿. {color}/{base} 치환 (미지정 시 task 기본값)",
    )
    parser.add_argument(
        "--base-label", type=str, default="base",
        help="stack instruction 의 {base} 라벨 (기본: 'base')",
    )
    parser.add_argument(
        "--keep-failed", action="store_true",
        help="실패한 episode도 저장한다 (기본: 실패 시 폴더 삭제)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="기본 RNG seed. worker w는 seed+w를 사용한다 (기본: None)",
    )
    parser.add_argument(
        "--camera-jitter", type=float, default=0.0,
        help="episode마다 카메라 위치를 ±이 값(미터)만큼 축별 균등 흔들어 시점을 다양화한다. "
             "0이면 XML 고정 시점 유지(기본: 0.0)",
    )
    parser.add_argument(
        "--speed", type=float, default=150,
        help="모션 속도(%%). joint_velocities = (speed/100)·MAX_SPEEDS (기본: 150)",
    )
    parser.add_argument(
        "--hz", type=float, default=10,
        help="프레임 로깅 주파수. 높일수록 프레임당 EE 델타가 작아진다 (기본: 10)",
    )
    parser.add_argument(
        "--settle-seconds", type=float, default=0.8,
        help="액션(웨이포인트)당 정착 시간(초). settle_seconds_per_action (기본: 0.8)",
    )
    parser.add_argument(
        "--noslip-iterations", type=int, default=None,
        help="마찰 잔류 slip 을 줄이는 후처리 PGS 반복 횟수. XML <option noslip_iterations> 를 "
             "런타임에 덮어쓴다. 낮출수록 스텝당 물리 비용↓(빠름)이나 그래스핑 중 큐브 slip↑. "
             "미지정 시 XML 값 유지 (기본: None)",
    )
    return parser.parse_args()


def _resolve_num_workers(spec, num_episodes):
    """'auto' → min(os.cpu_count(), num_episodes) 로 과구독을 방지. 정수면 그대로."""
    if isinstance(spec, str) and spec.lower() == "auto":
        cores = os.cpu_count() or 1
        resolved = max(1, min(cores, num_episodes))
        print(f"[num-workers] auto → {resolved} "
              f"(cpu_count={os.cpu_count()}, num_episodes={num_episodes})")
        return resolved
    try:
        n = int(spec)
    except (TypeError, ValueError):
        raise SystemExit(f"--num-workers 는 정수 또는 'auto' 여야 합니다: {spec!r}")
    if n < 1:
        raise SystemExit("--num-workers 는 1 이상이어야 합니다.")
    return n


if __name__ == "__main__":
    args = _parse_args()
    defaults = TASK_DEFAULTS[args.task]
    xml_path = args.xml_path or defaults["xml_path"]
    dataset_root = args.dataset_root or defaults["dataset_root"]
    instruction_template = args.instruction_template or defaults["instruction_template"]

    collect_dataset_parallel(
        num_workers=_resolve_num_workers(args.num_workers, args.num_episodes),
        xml_path=xml_path,
        dataset_root=dataset_root,
        num_episodes=args.num_episodes,
        colors=("red", "blue", "green", "yellow"),
        instruction_template=instruction_template,
        keep_failed=args.keep_failed,
        camera_name="front_view",
        camera_jitter=args.camera_jitter,
        speed=args.speed,
        hz=args.hz,
        settle_seconds_per_action=args.settle_seconds,
        noslip_iterations=args.noslip_iterations,
        initial_settle_seconds=0.1,
        seed=args.seed,
        object_x_range=(-0.10, 0.10),
        object_y_range=(0.16, 0.25),
        min_object_distance=0.035,
        task=args.task,
        base_label=args.base_label,
        episode_id_offset_base=defaults["episode_id_offset_base"],
    )


def _legacy_single_process_main():
    collect_dataset(
        xml_path="Raccoon_colored_cylinder.xml",
        dataset_root="raccoon_grasp_colored_cylinder",
        num_episodes=400,
        colors=("red", "blue", "green", "yellow"),
        instruction_template="grasp the {color} cylinder",
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        initial_settle_seconds=0.1,
        object_x_range=(-0.10, 0.10),
        object_y_range=(0.16, 0.25),
        min_object_distance=0.035,
    )
