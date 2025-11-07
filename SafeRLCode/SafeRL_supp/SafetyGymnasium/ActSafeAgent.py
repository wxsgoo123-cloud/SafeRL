# actsafe_agent.py
# Minimal, practical ActSafe baseline implemented "like CPO" for drop-in use
# with your existing SafeGym.ipynb training loop.
#
# Core idea (per paper/code): learn a probabilistic cost model and enforce
# pessimism at action time (shielding): only execute actions a such that
#   E[c(s,a)] + beta * Std[c(s,a)] <= eps_risk        (per-step)
# When violated, we project the policy's action to the nearest point in a
# linearized safe half-space around the proposed action.
#
# This file mirrors the structure of cpo_agent_safepo_style.py:
# - GaussianTanhPolicy & value MLP
# - Agent class with .act(...) and .update(...)
# Differences:
# - No CMDP optimization in the update step (we train on reward only).
# - Safety is enforced by an ActSafe-style action shield built from an
#   ensemble cost model (pessimistic confidence bound).
# - The cost model is trained from rollout data each update if per-step
#   costs are provided in 'data'.
#
# You can import and use exactly like your CPOAgent:
#   from actsafe_agent import ActSafeAgent, ActSafeConfig
#
# Expected 'data' keys for update (same as CPO where applicable):
#   obs [N,obs_dim], act [N,act_dim], logp [N], adv_r [N], ret_r [N],
#   (optional) cost [N]  <-- per-step safety costs for training the shield
#
# Notes:
# - The shield computes gradients w.r.t. actions using autograd on the
#   ensemble mean cost; for stability we ignore grad of the std term in
#   the projection step (common approximation).
# - If the proposed action is unsafe, we project with a single analytic
#   step to the nearest point in the linearized safe set (half-space).
# - If projection fails (degenerate), we softly shrink the action.
#


import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Small utils ----------
def to_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


# ---------- Networks ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64, 64), activation=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden, activation=nn.Tanh)
        # State-independent diagonal std (as in your CPO file)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def _atanh(self, y, eps=1e-6):
        y = torch.clamp(y, -1 + eps, 1 - eps)
        return 0.5 * torch.log((1 + y) / (1 - y))

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        # Tanh-corrected log prob
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - a.pow(2) + 1e-8).sum(-1)
        return a, logp

    def log_prob(self, obs, act):
        mu, std = self.forward(obs)
        x = self._atanh(act)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - act.pow(2) + 1e-8).sum(-1)
        return logp


def value_mlp(obs_dim, hidden=(64, 64)):
    return MLP(obs_dim, 1, hidden, activation=nn.Tanh)


# ---------- Ensemble Cost Model (for pessimistic safety) ----------
class EnsembleHead(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128)):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden, activation=nn.SiLU)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnsembleCostModel(nn.Module):
    """Bootstrapped ensemble to approximate E[c(s,a)] and epistemic uncertainty via disagreement."""
    def __init__(self, obs_dim, act_dim, n=5, hidden=(128, 128), lr=1e-3, weight_decay=1e-4, device='cpu'):
        super().__init__()
        self.n = n
        self.device = device
        self.heads = nn.ModuleList([EnsembleHead(obs_dim + act_dim, hidden) for _ in range(n)])
        self.opts = [torch.optim.Adam(h.parameters(), lr=lr, weight_decay=weight_decay) for h in self.heads]

    @torch.no_grad()
    def predict(self, s, a) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) across ensemble."""
        x = torch.cat([s, a], dim=-1)
        preds = torch.stack([h(x) for h in self.heads], dim=0)  # [n, B]
        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=False)
        return mean, std

    def loss(self, s, a, c):
        x = torch.cat([s, a], dim=-1)
        losses = []
        for h in self.heads:
            pred = h(x)
            losses.append(F.mse_loss(pred, c))
        return losses

    def step(self, s, a, c, iters=200, batch_size=1024):
        s, a, c = to_t(s, self.device), to_t(a, self.device), to_t(c, self.device)
        n = s.shape[0]
        if n == 0:
            return {}
        idx = torch.arange(n, device=self.device)
        for _ in range(iters):
            # Bootstrap via resampling each head independently
            for h, opt in zip(self.heads, self.opts):
                samp = idx[torch.randint(0, n, (min(batch_size, n),), device=self.device)]
                xs, xa, yc = s[samp], a[samp], c[samp]
                opt.zero_grad()
                loss = F.mse_loss(h(torch.cat([xs, xa], dim=-1)).squeeze(-1), yc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 10.0)
                opt.step()
        return {}


# ---------- ActSafe Shield (pessimistic constraint) ----------
@dataclass
class ShieldConfig:
    eps_risk: float = 0.0     # per-step safe threshold, e.g., hazard indicator
    beta: float = 2.0         # pessimism factor on uncertainty
    max_proj_steps: int = 1   # single analytic projection (linear half-space)
    shrink_on_fail: float = 0.8  # fallback scaling if projection degenerates
    grad_eps: float = 1e-8


class ActSafeShield:
    def __init__(self, cost_model: EnsembleCostModel, cfg: ShieldConfig, act_low=-1.0, act_high=1.0, device='cpu'):
        self.cost_model = cost_model
        self.cfg = cfg
        self.device = device
        self.act_low = to_t(act_low, device)
        self.act_high = to_t(act_high, device)

    def _pess_cost(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pessimistic bound, mean, std."""
        mean, std = self.cost_model.predict(s, a)
        pess = mean + self.cfg.beta * std
        return pess, mean, std

    def project(self, s: torch.Tensor, a0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project action onto linearized safe half-space:
           minimize ||a - a0||^2 s.t. mean_c(s,a0) + beta*std(s,a0) + grad_mean(a0)^T (a - a0) <= eps
           (Ignore grad of std for stability.)
        """
        s = s.detach().requires_grad_(False).to(self.device)
        a0 = a0.detach().to(self.device)
        a = a0.clone()

        # Compute mean and std at a0; grad wrt action of mean only
        a_req = a0.clone().detach().requires_grad_(True)
        mean, std = self.cost_model.predict(s, a_req)
        pess = mean + self.cfg.beta * std

        unsafe = (pess > self.cfg.eps_risk)
        info = {"unsafe": unsafe.item() if unsafe.numel() == 1 else unsafe.cpu().numpy()}
        if not unsafe.any():
            return torch.clamp(a0, self.act_low, self.act_high), {**info, "projected": False}

        # grad of mean wrt action
        grad_mean = torch.autograd.grad(mean.sum(), a_req, retain_graph=False, create_graph=False)[0]  # [act_dim]
        g = grad_mean  # treat as row vector in constraint g^T (a - a0) <= eps - (mean + beta*std)

        # linear RHS
        rhs = self.cfg.eps_risk - (mean + self.cfg.beta * std)  # scalar (or batch)
        # If g is near zero, fall back to shrink
        denom = (g * g).sum() + self.cfg.grad_eps
        # Analytic projection to nearest point on half-space boundary:
        # a* = a0 + ((rhs) / ||g||^2) * g
        step = (rhs / denom) * g
        a_new = a0 + step
        a_new = torch.clamp(a_new, self.act_low, self.act_high)

        # Check feasibility after projection (recompute pess bound)
        pess_new, _, _ = self._pess_cost(s, a_new)
        if (pess_new <= self.cfg.eps_risk).all():
            return a_new.detach(), {**info, "projected": True, "method": "halfspace"}

        # Fallback: shrink toward zero (or to a0 scaled)
        a_fallback = torch.clamp(self.cfg.shrink_on_fail * a0, self.act_low, self.act_high)
        return a_fallback.detach(), {**info, "projected": True, "method": "shrink"}


# ---------- PPO-style update (lightweight, stable) ----------
def ppo_clip_loss(pi: GaussianTanhPolicy, obs, act, logp_old, adv, clip_ratio=0.2):
    logp = pi.log_prob(obs, act)
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    pi_loss = -torch.mean(torch.min(unclipped, clipped))
    approx_kl = torch.mean(logp_old - logp).clamp(min=0)  # non-negative
    return pi_loss, approx_kl, logp


# ---------- Configs ----------
@dataclass
class ActSafeConfig:
    # on-policy update settings
    vf_lr: float = 3e-4
    pi_lr: float = 3e-4
    vf_iters: int = 80
    pi_iters: int = 80
    clip_ratio: float = 0.2
    target_kl: float = 0.01

    # shield / model settings
    shield_beta: float = 2.0
    shield_eps_risk: float = 0.0
    shield_shrink_on_fail: float = 0.8
    ensemble_size: int = 5
    ensemble_lr: float = 1e-3
    ensemble_wd: float = 1e-4
    ensemble_hidden: Tuple[int, int] = (128, 128)
    model_train_iters: int = 200
    model_batch_size: int = 1024
    cost_limit=5
    # misc
    action_low: float = -1.0
    action_high: float = 1.0


# ---------- ActSafe Agent ----------
class ActSafeAgent:
    """
    ActSafe baseline agent:
      - Policy/value trained on reward using PPO-style update
      - Pessimistic action shield using an ensemble cost model
    """
    def __init__(self, obs_space, act_space, cfg: ActSafeConfig, device='cpu'):
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        self.cfg = cfg
        self.device = device

        # Policy / value
        self.pi = GaussianTanhPolicy(obs_dim, act_dim).to(device)
        self.vf = value_mlp(obs_dim).to(device)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.pi_lr)
        self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=cfg.vf_lr)

        # Ensemble cost model + shield
        self.cost_model = EnsembleCostModel(
            obs_dim, act_dim, n=cfg.ensemble_size,
            hidden=cfg.ensemble_hidden, lr=cfg.ensemble_lr,
            weight_decay=cfg.ensemble_wd, device=device
        ).to(device)
        shield_cfg = ShieldConfig(
            eps_risk=cfg.shield_eps_risk,
            beta=cfg.shield_beta,
            shrink_on_fail=cfg.shield_shrink_on_fail
        )
        self.shield = ActSafeShield(
            self.cost_model, shield_cfg,
            act_low=cfg.action_low, act_high=cfg.action_high, device=device
        )

        self._last_info = {}

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Sample action from policy, then apply ActSafe shield.
        Returns (action, logp_under_policy_of_final_action, info)."""
        obs = to_t(obs, self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        a, _ = self.pi.sample(obs)
        a_shielded, s_info = self.shield.project(obs, a)
        # Compute logp of the final (possibly projected) action under current policy
        logp = self.pi.log_prob(obs, a_shielded)

        info = {"shielded": bool(s_info.get("projected", False)), **s_info}
        self._last_info = info
        return a_shielded.squeeze(0), logp.squeeze(0), info

    def update(self, data: Dict[str, torch.Tensor], ep_cost_mean: float, avg_ep_len: float) -> Dict[str, Any]:
        """
        Training step using on-policy data (like your CPO agent).
        data keys used:
          obs [N,obs_dim], act [N,act_dim], logp [N], adv_r [N], ret_r [N]
          (optional) cost [N]  -> trains the ensemble cost model
        """
        obs = to_t(data['obs'], self.device)
        act = to_t(data['act'], self.device)
        logp_old = to_t(data['logp'], self.device)
        adv_r = to_t(data['adv_r'], self.device)
        ret_r = to_t(data['ret_r'], self.device)

        # Normalize advantages (optional but common)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

        # --- Policy update (PPO-style) ---
        pi_info = {}
        for i in range(self.cfg.pi_iters):
            self.pi_opt.zero_grad()
            pi_loss, approx_kl, logp = ppo_clip_loss(self.pi, obs, act, logp_old, adv_r, self.cfg.clip_ratio)
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 10.0)
            self.pi_opt.step()
            if approx_kl.item() > 1.5 * self.cfg.target_kl:
                pi_info["early_stop_iter"] = i + 1
                break
        pi_info.update({"pi_loss": float(pi_loss.item()), "kl": float(approx_kl.item())})

        # --- Value function update ---
        for _ in range(self.cfg.vf_iters):
            v_loss = F.mse_loss(self.vf(obs).squeeze(-1), ret_r)
            self.vf_opt.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 10.0)
            self.vf_opt.step()

        # --- Train / refresh ensemble cost model (if per-step costs provided) ---
        model_info = {}
        if 'cost' in data:
            c = to_t(data['cost'], self.device)
            self.cost_model.step(
                obs, act, c,
                iters=self.cfg.model_train_iters,
                batch_size=self.cfg.model_batch_size
            )
            # Report fit quality on the batch
            with torch.no_grad():
                mean_c, std_c = self.cost_model.predict(obs, act)
                mse = F.mse_loss(mean_c, c).item()
                coverage = (mean_c + self.cfg.shield_beta * std_c <= self.cfg.shield_eps_risk).float().mean().item()
            model_info = {"cost_mse": mse, "pess_cov_eps": coverage}

        # --- Log shield stats from the last acting calls (if any) ---
        shield_info = {
            "shield_last_projected": bool(self._last_info.get("projected", False)),
            "shield_last_method": self._last_info.get("method", "n/a"),
        }

        return {
            **pi_info,
            **model_info,
            **shield_info,
            "ep_cost_mean": float(ep_cost_mean),
            "avg_ep_len": float(avg_ep_len),
        }

