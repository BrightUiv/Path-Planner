from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

IS_LINUX = sys.platform.startswith("linux")
if IS_LINUX:
    # 服务器上禁用渲染，避免 OpenGL 依赖
    os.environ.setdefault("GS_DISABLE_RENDERING", "1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("PYGLET_HEADLESS", "1")
else:
    # Windows/macOS 不支持 OSMesa，避免强制加载
    os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.pop("PYGLET_HEADLESS", None)
    # Windows 下禁用 ndarray 模式，避免 ScalarNdarray 的 DLPack 兼容问题
    os.environ.setdefault("GS_ENABLE_NDARRAY", "0")

import genesis as gs

from env_genesis import SingleDroneDynAvoidEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu", "cuda"])
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()

    backend = getattr(gs, args.backend)
    gs.init(backend=backend, precision="32", logging_level="warning")

    env = SingleDroneDynAvoidEnv(n_envs=1, backend=backend, show_viewer=args.vis)
    obs = env.reset()

    action = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=torch.float32)
    for _ in range(args.steps):
        obs, rew, done, info = env.step(action)
        if bool(done.item()):
            obs = env.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
