from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from genesis.utils.geom import quat_to_xyz

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity


@dataclass
class RPMClamp:
    base_rpm: float = 14468.429183500699
    min_scale: float = 0.9
    max_scale: float = 1.5

    @property
    def min_rpm(self) -> float:
        return self.min_scale * self.base_rpm

    @property
    def max_rpm(self) -> float:
        return self.max_scale * self.base_rpm

    def clamp(self, rpm: float) -> int:
        return int(max(self.min_rpm, min(rpm, self.max_rpm)))


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class DronePIDController:
    """
    使用位置/速度/姿态 PID，把目标位置转成四旋翼电机 rpm。
    """

    def __init__(self, drone: "DroneEntity", dt: float, base_rpm: float, pid_params: list[list[float]]):
        self._pid_pos_x = PIDController(*pid_params[0])
        self._pid_pos_y = PIDController(*pid_params[1])
        self._pid_pos_z = PIDController(*pid_params[2])

        self._pid_vel_x = PIDController(*pid_params[3])
        self._pid_vel_y = PIDController(*pid_params[4])
        self._pid_vel_z = PIDController(*pid_params[5])

        self._pid_att_roll = PIDController(*pid_params[6])
        self._pid_att_pitch = PIDController(*pid_params[7])
        self._pid_att_yaw = PIDController(*pid_params[8])

        self.drone = drone
        self._dt = float(dt)
        self._base_rpm = float(base_rpm)

    def _get_pos(self) -> torch.Tensor:
        pos = self.drone.get_pos()
        if pos.ndim == 2 and pos.shape[0] == 1:
            pos = pos[0]
        return pos

    def _get_vel(self) -> torch.Tensor:
        vel = self.drone.get_vel()
        if vel.ndim == 2 and vel.shape[0] == 1:
            vel = vel[0]
        return vel

    def _get_att_rpy_deg(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        rpy = quat_to_xyz(quat, rpy=True, degrees=True)
        if rpy.ndim == 2 and rpy.shape[0] == 1:
            rpy = rpy[0]
        return rpy

    def _mixer(self, thrust: float, roll: float, pitch: float, yaw: float, x_vel: float, y_vel: float) -> torch.Tensor:
        m1 = self._base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        m2 = self._base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        m3 = self._base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        m4 = self._base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.tensor([m1, m2, m3, m4], dtype=torch.float32)

    def update(self, target_pos_xyz: torch.Tensor) -> np.ndarray:
        curr_pos = self._get_pos()
        curr_vel = self._get_vel()
        curr_att = self._get_att_rpy_deg()
        if curr_pos.ndim != 1:
            raise ValueError("DronePIDController expects a single drone state (shape (3,)).")

        err_pos_x = float(target_pos_xyz[0] - curr_pos[0])
        err_pos_y = float(target_pos_xyz[1] - curr_pos[1])
        err_pos_z = float(target_pos_xyz[2] - curr_pos[2])

        vel_des_x = self._pid_pos_x.update(err_pos_x, self._dt)
        vel_des_y = self._pid_pos_y.update(err_pos_y, self._dt)
        vel_des_z = self._pid_pos_z.update(err_pos_z, self._dt)

        err_vel_x = float(vel_des_x - curr_vel[0])
        err_vel_y = float(vel_des_y - curr_vel[1])
        err_vel_z = float(vel_des_z - curr_vel[2])

        x_vel_delta = self._pid_vel_x.update(err_vel_x, self._dt)
        y_vel_delta = self._pid_vel_y.update(err_vel_y, self._dt)
        thrust_des = self._pid_vel_z.update(err_vel_z, self._dt)

        err_roll = float(0.0 - curr_att[0])
        err_pitch = float(0.0 - curr_att[1])
        err_yaw = float(0.0 - curr_att[2])

        roll_delta = self._pid_att_roll.update(err_roll, self._dt)
        pitch_delta = self._pid_att_pitch.update(err_pitch, self._dt)
        yaw_delta = self._pid_att_yaw.update(err_yaw, self._dt)

        rpms = self._mixer(thrust_des, roll_delta, pitch_delta, yaw_delta, x_vel_delta, y_vel_delta)
        return rpms.cpu().numpy()
