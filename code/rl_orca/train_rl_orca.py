from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

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
from policy import ActorCritic


@dataclass
class TrainCfg:
    steps_per_update: int = 256
    total_updates: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


def compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, gamma: float, lam: float):
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    next_value = next_value.squeeze()
    last = torch.zeros_like(rewards[0])
    for i in reversed(range(t)):
        mask = 1.0 - dones[i]
        delta = rewards[i] + gamma * (next_value if i == t - 1 else values[i + 1]) * mask - values[i]
        last = delta + gamma * lam * mask * last
        adv[i] = last
    returns = adv + values
    return adv, returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu", "cuda"])
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--total_updates", type=int, default=200)
    parser.add_argument("--steps_per_update", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_to", type=str, default=str(REPO_ROOT / "code/rl_orca/rl_orca_policy.pt"))
    parser.add_argument("--eval_only", action="store_true", default=False)
    args = parser.parse_args()

    cfg = TrainCfg(steps_per_update=args.steps_per_update, total_updates=args.total_updates)

    backend = getattr(gs, args.backend)
    gs.init(backend=backend, precision="32", logging_level="warning")

    env = SingleDroneDynAvoidEnv(n_envs=1, backend=backend, show_viewer=args.vis)
    ac = ActorCritic(obs_dim=env.obs_dim, act_dim=env.act_dim).to(gs.device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=gs.device)
        ac.load_state_dict(state["model"])

    if args.eval_only:
        obs = env.reset()
        for _ in range(4000):
            out = ac.act(obs)
            obs, rew, done, info = env.step(out.action)
            if bool(done.item()):
                obs = env.reset()
        return 0

    obs = env.reset()
    for update in range(cfg.total_updates):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        ent_buf = []

        for _ in range(cfg.steps_per_update):
            out = ac.act(obs)
            next_obs, rew, done, info = env.step(out.action)

            obs_buf.append(obs)
            act_buf.append(out.action)
            logp_buf.append(out.logp)
            val_buf.append(out.value)
            ent_buf.append(out.entropy)
            rew_buf.append(rew)
            done_buf.append(done)

            obs = next_obs
            if bool(done.item()):
                obs = env.reset()

        obs_t = torch.cat(obs_buf, dim=0)
        act_t = torch.cat(act_buf, dim=0)
        logp_t = torch.cat(logp_buf, dim=0)
        val_t = torch.cat(val_buf, dim=0)
        rew_t = torch.cat(rew_buf, dim=0)
        done_t = torch.cat(done_buf, dim=0)

        with torch.no_grad():
            _, next_v = ac(obs)
        adv, ret = compute_gae(rew_t, done_t, val_t, next_v, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        new_logp, new_v, entropy = ac.evaluate_actions(obs_t, act_t)
        policy_loss = -(adv * new_logp).mean()
        value_loss = F.mse_loss(new_v, ret)
        ent_loss = -entropy.mean()
        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * ent_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
        opt.step()

        if (update + 1) % 10 == 0:
            print(
                f"update={update+1:04d} loss={float(loss.item()):.3f} "
                f"policy={float(policy_loss.item()):.3f} value={float(value_loss.item()):.3f} ent={float(entropy.mean().item()):.3f}"
            )

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ac.state_dict()}, args.save_to)
    print(f"saved: {args.save_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
