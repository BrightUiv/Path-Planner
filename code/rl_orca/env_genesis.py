from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import genesis as gs

from quadcopter_controller import DronePIDController, RPMClamp
from orca import OrcaParams, vo_orca_velocity
from policy import OrcaParamBounds, decode_orca_params


@dataclass
class ObstacleTraj:
    base_pos: torch.Tensor  # (3,)
    axis: torch.Tensor  # (3,)
    amp: float
    omega: float
    radius: float
    phase: torch.Tensor  # (B,)

    def pos_vel(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        arg = self.phase + self.omega * t
        s = torch.sin(arg)[:, None]
        c = torch.cos(arg)[:, None]
        axis = self.axis[None, :].to(self.phase.device)
        base = self.base_pos[None, :].to(self.phase.device)
        pos = base + axis * (self.amp * s)
        vel = axis * (self.amp * self.omega * c)
        return pos, vel


class SingleDroneDynAvoidEnv:
    """
    单架无人机 + 动态障碍物（多并行 env 可选）。

    动作：4 维连续值（[-1,1]），用于调 ORCA 关键参数：
      - time_horizon / safety_margin / avoid_weight / max_speed
    控制：ORCA 输出期望速度 -> 转成小步目标位置 -> PID 输出电机 rpm。
    """

    def __init__(
        self,
        n_envs: int = 1,
        *,
        backend: gs.constants.backend = gs.cpu,
        dt: float = 0.01,
        substeps: int = 2,
        episode_len_s: float = 12.0,
        show_viewer: bool = False,
        n_obstacles: int = 3,
        seed: int = 0,
    ):
        self.n_envs = int(n_envs)
        self.dt = float(dt)
        self.max_steps = int(round(episode_len_s / dt))
        self.device = gs.device

        self.bounds = OrcaParamBounds()
        self._t = 0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=int(substeps)),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=bool(show_viewer),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(5.0, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=50,
            )
            if show_viewer
            else None,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, -2.0, 0.25)))

        self.goal = torch.tensor([0.0, 2.0, 0.25], dtype=gs.tc_float, device=self.device)[None, :].repeat(
            self.n_envs if self.n_envs > 0 else 1, 1
        )

        torch.manual_seed(int(seed))
        self._obstacles: list[dict[str, Any]] = []
        self._traj: list[ObstacleTraj] = []
        for i in range(int(n_obstacles)):
            base = torch.tensor(
                [(-1.0 + 2.0 * i / max(1, n_obstacles - 1)), 0.0, 0.35],
                dtype=gs.tc_float,
                device=self.device,
            )
            axis = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device)
            amp = 0.7
            omega = 1.0 + 0.4 * i
            radius = 0.18
            phase = (2.0 * torch.pi) * torch.rand(
                (self.n_envs if self.n_envs > 0 else 1,), device=self.device, dtype=gs.tc_float
            )
            traj = ObstacleTraj(base_pos=base, axis=axis, amp=amp, omega=omega, radius=radius, phase=phase.to(self.device))

            ent = self.scene.add_entity(
                morph=gs.morphs.Sphere(pos=tuple(base.tolist()), radius=radius, fixed=True, collision=True),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.3, 0.3))),
            )
            self._obstacles.append({"entity": ent, "radius": radius})
            self._traj.append(traj)

        if show_viewer:
            self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.08, pos=tuple(self.goal[0].tolist()), fixed=True),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.9, 0.3))),
            )

        self.scene.build(n_envs=self.n_envs)

        pid_params = [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [20.0, 0.0, 20.0],
            [20.0, 0.0, 20.0],
            [25.0, 0.0, 20.0],
            [10.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [2.0, 0.0, 0.2],
        ]
        if self.n_envs != 1:
            raise ValueError("This minimal controller wrapper is implemented for n_envs=1 to keep code short.")
        self._rpm_clamp = RPMClamp()
        self._controller = DronePIDController(
            drone=self.drone, dt=self.dt, base_rpm=self._rpm_clamp.base_rpm, pid_params=pid_params
        )

        self.obs_dim = 3 + 3 + 3 + 2 * 6
        self.act_dim = 4

        self.drone_radius = 0.18
        self.collision_dist = 0.35
        self.goal_dist = 0.25

    def reset(self) -> torch.Tensor:
        self._t = 0
        self.scene.reset()
        self.drone.set_pos((0.0, -2.0, 0.25))
        self.drone.set_quat((1.0, 0.0, 0.0, 0.0))
        rpm = torch.full((1, 4), self._rpm_clamp.base_rpm, device=self.device, dtype=gs.tc_float)
        self.drone.set_propellels_rpm(rpm)
        # 让首个 step 不触发“同一仿真步重复设转速”的保护
        if hasattr(self.drone, "_prev_prop_t"):
            self.drone._prev_prop_t = None
        self._set_obstacles()
        return self._get_obs()

    def _set_obstacles(self) -> None:
        t = self._t * self.dt
        for ent_info, traj in zip(self._obstacles, self._traj, strict=True):
            pos, _vel = traj.pos_vel(t)
            ent_info["entity"].set_pos(pos[0], zero_velocity=True)

    def _get_obstacles_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self._t * self.dt
        pos_list, vel_list, r_list = [], [], []
        for ent_info, traj in zip(self._obstacles, self._traj, strict=True):
            pos, vel = traj.pos_vel(t)
            pos_list.append(pos)
            vel_list.append(vel)
            r_list.append(torch.tensor(ent_info["radius"], dtype=gs.tc_float, device=self.device))
        pos_t = torch.stack(pos_list, dim=1)  # (B,K,3)
        vel_t = torch.stack(vel_list, dim=1)
        r_t = torch.stack(r_list, dim=0)  # (K,)
        return pos_t, vel_t, r_t

    def _get_obs(self) -> torch.Tensor:
        pos = self.drone.get_pos()[0].unsqueeze(0)
        vel = self.drone.get_vel()[0].unsqueeze(0)
        goal_vec = self.goal - pos

        obs_pos, obs_vel, _r = self._get_obstacles_state()
        rel_pos = obs_pos - pos[:, None, :]
        rel_vel = obs_vel - vel[:, None, :]

        k = min(2, rel_pos.shape[1])
        rel_pos = rel_pos[:, :k, :]
        rel_vel = rel_vel[:, :k, :]

        feat = torch.cat(
            [
                goal_vec,
                vel,
                pos,
                rel_pos.reshape(rel_pos.shape[0], -1),
                rel_vel.reshape(rel_vel.shape[0], -1),
            ],
            dim=-1,
        )
        return feat.to(dtype=torch.float32)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action = action.detach()
        params_t = decode_orca_params(action, self.bounds)

        pos = self.drone.get_pos()[0]
        vel = self.drone.get_vel()[0]
        goal_vec = self.goal[0] - pos

        goal_dir = goal_vec / (goal_vec.norm() + 1e-6)
        pref_vel = goal_dir[None, :] * params_t["max_speed"][0].clamp(min=0.1).item()

        obs_pos, obs_vel, obs_r = self._get_obstacles_state()
        params = OrcaParams(
            time_horizon=float(params_t["time_horizon"][0].item()),
            safety_margin=float(params_t["safety_margin"][0].item()),
            avoid_weight=float(params_t["avoid_weight"][0].item()),
            max_speed=float(params_t["max_speed"][0].item()),
        )

        v_des = vo_orca_velocity(
            drone_pos=pos.unsqueeze(0),
            drone_vel=vel.unsqueeze(0),
            pref_vel=pref_vel,
            obs_pos=obs_pos,
            obs_vel=obs_vel,
            drone_radius=self.drone_radius,
            obs_radius=obs_r,
            params=params,
            dt=self.dt,
        )[0]
        v_des[2] = 0.0

        target_pos = pos + v_des * self.dt
        rpms = self._controller.update(target_pos)
        rpms = [self._rpm_clamp.clamp(float(r)) for r in rpms]
        rpm = torch.tensor(rpms, device=self.device, dtype=gs.tc_float).unsqueeze(0)
        if hasattr(self.drone, "_prev_prop_t"):
            self.drone._prev_prop_t = None
        self.drone.set_propellels_rpm(rpm)

        self._set_obstacles()
        self.scene.step()
        self._t += 1

        obs = self._get_obs()
        rew, done, info = self._reward_done_info()
        return obs, rew, done, info

    def _reward_done_info(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        pos = self.drone.get_pos()[0]
        dist_goal = (self.goal[0] - pos).norm()

        obs_pos, _obs_vel, _obs_r = self._get_obstacles_state()
        dists = (obs_pos[0] - pos[None, :]).norm(dim=-1)
        min_dist = dists.min()

        collided = min_dist < self.collision_dist
        reached = dist_goal < self.goal_dist
        timeout = self._t >= self.max_steps

        reward = 0.0
        reward += float(-0.2 * dist_goal)
        reward += 0.05
        reward += float(0.4 * torch.tanh((min_dist - self.collision_dist) * 2.0))
        if collided:
            reward -= 5.0
        if reached:
            reward += 8.0

        done = collided or reached or timeout
        info = {
            "dist_goal": float(dist_goal.item()),
            "min_obs_dist": float(min_dist.item()),
            "collided": bool(collided),
            "reached": bool(reached),
            "timeout": bool(timeout),
        }
        return (
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            torch.tensor([1 if done else 0], dtype=torch.float32, device=self.device),
            info,
        )
