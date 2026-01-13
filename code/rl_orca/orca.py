from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OrcaParams:
    """
    这些参数是 RL 的动作空间“可学习/可调”的目标：
    - `time_horizon`: 动态障碍物预测时域（越大越保守）。
    - `safety_margin`: 额外安全裕度（越大越保守）。
    - `avoid_weight`: 避障修正强度（越大越偏离最短路）。
    - `max_speed`: 速度上限（越大越快但更易碰撞/抖动）。
    """

    time_horizon: float = 1.2
    safety_margin: float = 0.12
    avoid_weight: float = 1.0
    max_speed: float = 1.0


def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))


def vo_orca_velocity(
    drone_pos: torch.Tensor,
    drone_vel: torch.Tensor,
    pref_vel: torch.Tensor,
    obs_pos: torch.Tensor,
    obs_vel: torch.Tensor,
    drone_radius: float,
    obs_radius: torch.Tensor,
    params: OrcaParams,
    dt: float,
) -> torch.Tensor:
    """
    简化版 VO/ORCA：基于“最近接距离 < 合并半径”的条件，在速度空间做修正。

    Shapes
    - `drone_pos/drone_vel/pref_vel`: (B, 3)
    - `obs_pos/obs_vel`: (B, K, 3)
    - `obs_radius`: (K,) 或 (B, K)
    """
    if drone_pos.ndim != 2 or obs_pos.ndim != 3:
        raise ValueError("Expected drone tensors (B,3) and obstacle tensors (B,K,3).")

    bsz, k, _ = obs_pos.shape

    rel_pos = obs_pos - drone_pos[:, None, :]
    rel_vel = drone_vel[:, None, :] - obs_vel

    rel_speed2 = (rel_vel * rel_vel).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    ttc = -(rel_pos * rel_vel).sum(dim=-1, keepdim=True) / rel_speed2
    ttc = torch.clamp(ttc, 0.0, float(params.time_horizon))

    closest = rel_pos + rel_vel * ttc
    closest_dist = _safe_norm(closest)  # (B,K,1)

    combined = drone_radius + params.safety_margin + obs_radius
    if combined.ndim == 1:
        combined = combined[None, :, None].expand(bsz, k, 1)
    elif combined.ndim == 2:
        combined = combined[:, :, None]
    else:
        combined = combined

    penetration = (combined - closest_dist).clamp_min(0.0)  # (B,K,1)
    push_dir = -closest / closest_dist

    denom = torch.clamp(ttc, min=dt)
    dv = push_dir * (penetration / denom)  # (B,K,3)
    dv = dv.sum(dim=1) * float(params.avoid_weight)  # (B,3)

    v = pref_vel + dv
    speed = _safe_norm(v)
    max_speed = float(params.max_speed)
    if max_speed > 0.0:
        v = v * torch.clamp(max_speed / speed, max=1.0)
    return v
