import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


# ---------- Utilities to flatten params/grads ----------
def flat_params(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in m.parameters()])


def set_params(m: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n


def flat_grad(y: torch.Tensor, model: nn.Module, retain_graph=False, create_graph=False) -> torch.Tensor:
    grads = torch.autograd.grad(
        y, [p for p in model.parameters() if p.requires_grad],
        retain_graph=retain_graph, create_graph=create_graph, allow_unused=False
    )
    return torch.cat([g.contiguous().view(-1) for g in grads])


def cg_solve(f_Ax, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        Avp = f_Ax(p)
        denom = torch.dot(p, Avp) + 1e-8
        alpha = rdotr / denom
        x = x + alpha * p
        r = r - alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


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
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, obs):
        dist = torch.distributions.Normal(*self.forward(obs))
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - a.pow(2) + 1e-8).sum(-1)
        return a, logp

    def log_prob(self, obs, act):
        mu, std = self.forward(obs)
        eps = 1e-6
        x = 0.5 * torch.log((1 + act + eps) / (1 - act + eps))  # atanh(act)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - act.pow(2) + 1e-8).sum(-1)
        return logp


def value_mlp(obs_dim, hidden=(64, 64)):
    return MLP(obs_dim, 1, hidden, activation=nn.Tanh)


# ---------- CPO Agent (SafePO-style) ----------
@dataclass
class CPOConfig:
    steps_per_epoch: int = 20000
    epochs: int = 50
    gamma: float = 0.99
    lam: float = 0.95
    gamma_c: float = 1
    lam_c: float = 0.95
    max_kl: float = 0.01
    cg_iters: int = 30
    cg_damping: float = 0.1
    backtrack_iters: int = 50
    backtrack_ratio: float = 0.8
    vf_lr: float = 3e-4
    vf_iters: int = 80
    cost_vf_lr: float = 6e-4
    cost_vf_iters: int = 240
    cost_limit: float = 0.0


class CPOAgent:
    def __init__(self, obs_space, act_space, cfg: CPOConfig, device='cpu'):
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        self.cfg = cfg
        self.device = device

        self.pi = GaussianTanhPolicy(obs_dim, act_dim).to(device)
        self.pi_old = GaussianTanhPolicy(obs_dim, act_dim).to(device)
        self.vf = value_mlp(obs_dim).to(device)
        self.cvf = value_mlp(obs_dim).to(device)

        self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=cfg.vf_lr)
        self.cvf_opt = torch.optim.Adam(self.cvf.parameters(), lr=cfg.cost_vf_lr)

        self.pi_old.load_state_dict(self.pi.state_dict())

    # Fisher-vector product using KL(old||new)
    def _fvp(self, vec: torch.Tensor, obs: torch.Tensor, damping=None) -> torch.Tensor:
        if damping is None:
            damping = self.cfg.cg_damping

        mu, std = self.pi.forward(obs)
        mu_old, std_old = self.pi_old.forward(obs)

        dist = torch.distributions.Normal(mu, std)
        dist_old = torch.distributions.Normal(mu_old, std_old)

        kl = torch.distributions.kl_divergence(dist_old, dist).sum(-1).mean()
        grads = torch.autograd.grad(kl, self.pi.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])

        kl_v = (flat_grad_kl * vec).sum()
        grads2 = torch.autograd.grad(kl_v, self.pi.parameters(), retain_graph=True)
        fvp = torch.cat([g.contiguous().view(-1) for g in grads2])
        return fvp + damping * vec

    def _losses(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, act, logp_old = data['obs'], data['act'], data['logp']
        adv_r, adv_c = data['adv_r'], data['adv_c']
        ret_r, ret_c = data['ret_r'], data['ret_c']

        logp = self.pi.log_prob(obs, act)
        ratio = torch.exp(logp - logp_old)

        surr_r = (ratio * adv_r).mean()
        surr_c = (ratio * adv_c).mean()

        v_loss = ((self.vf(obs).squeeze(-1) - ret_r) ** 2).mean()
        c_loss = ((self.cvf(obs).squeeze(-1) - ret_c) ** 2).mean()
        return surr_r, surr_c, v_loss, c_loss, logp

    def update(self, data, ep_cost_mean: float, avg_ep_len: float) -> Dict[str, Any]:
        info: Dict[str, Any] = {}

        # Snapshot old policy for KL / FVP
        self.pi_old.load_state_dict(self.pi.state_dict())

        obs, act, logp_old = data['obs'], data['act'], data['logp']
        adv_r, adv_c = data['adv_r'], data['adv_c']
        ret_r, ret_c = data['ret_r'], data['ret_c']

        # Losses at current params
        surr_r, surr_c, v_loss, c_loss, _ = self._losses(data)

        # Gradients wrt actor params
        g = flat_grad(surr_r, self.pi, retain_graph=True)  # reward gradient
        b = flat_grad(surr_c, self.pi, retain_graph=True)  # cost gradient

        # Episode-level residual (paper-faithful scaling)
        d_ep = self.cfg.cost_limit * avg_ep_len if self.cfg.cost_limit <= 1.0 else self.cfg.cost_limit
        c_res = float(ep_cost_mean - d_ep)
        info['c_res'] = c_res

        # Natural directions via CG
        fvp = lambda v: self._fvp(v, obs)
        w = cg_solve(fvp, g, iters=self.cfg.cg_iters)       # H^{-1} g
        v_dir = cg_solve(fvp, b, iters=self.cfg.cg_iters)   # H^{-1} b

        # Quadratic forms
        q = torch.dot(g, w).item()        # w^T H w
        r = torch.dot(g, v_dir).item()    # g^T H^{-1} b = b^T H^{-1} g
        s = torch.dot(b, v_dir).item()    # b^T H^{-1} b
        info.update(dict(q=q, r=r, s=s))
        print({
        "c_res": c_res,
        "adv_c_mean": adv_c.mean().item(),
        "adv_c_std": adv_c.std().item()
        })
        # TRPO step along w under KL constraint
        if q > 0:
            step_trpo = (math.sqrt(2 * self.cfg.max_kl / (q + 1e-8)) * w).detach()
        else:
            step_trpo = torch.zeros_like(w)

        # Linearized feasibility of TRPO step
        b_dot_x_trpo = torch.dot(b, step_trpo).item()
        lin_feasible_trpo = (b_dot_x_trpo + c_res) <= 0.0

        # If not feasible, compute projected step a*w + b*v that meets linearized constraint and KL
        step_dir = step_trpo.clone()
        if not lin_feasible_trpo:
            B = r      # b^T H^{-1} g
            G = s      # b^T H^{-1} b
            if G <= 0 or math.isnan(G):
                # Fallback: feasible direction along v_dir
                if s > 0:
                    scale_v = math.sqrt(2 * self.cfg.max_kl / (s + 1e-8))
                    step_dir = (-scale_v) * v_dir
                else:
                    step_dir = torch.zeros_like(w)
            else:
                # Choose alpha solving the trust-region boundary; beta ensures linearized feasibility
                # beta(alpha) = (-c_res - B*alpha)/G
                A2 = (q - 2 * r * B / G + s * (B * B) / (G * G))
                A1 = (-2 * r * c_res / G + 2 * s * c_res * B / (G * G))
                A0 = (s * (c_res ** 2) / (G * G) - 2 * self.cfg.max_kl)

                disc = A1 * A1 - 4 * A2 * A0
                if A2 <= 0 or disc < 0:
                    # fall back to feasible v_dir
                    if s > 0:
                        scale_v = math.sqrt(2 * self.cfg.max_kl / (s + 1e-8))
                        step_dir = (-scale_v) * v_dir
                    else:
                        step_dir = torch.zeros_like(w)
                else:
                    sqrt_disc = math.sqrt(disc)
                    alpha_candidates = [(-A1 + sqrt_disc) / (2 * A2), (-A1 - sqrt_disc) / (2 * A2)]
                    best = None; best_val = -1e30
                    for alpha in alpha_candidates:
                        beta = (-c_res - B * alpha) / G
                        J = alpha * q + beta * r   # linearized reward improvement
                        if J > best_val:
                            best_val = J; best = (alpha, beta)
                    alpha, beta = best
                    step_dir = alpha * w + beta * v_dir

        # Line search: check true surrogate reward, true surrogate cost, and KL
        def eval_at(new_flat):
            old_flat = flat_params(self.pi).detach()
            set_params(self.pi, new_flat)
            surr_r_new, surr_c_new, _, _, _ = self._losses(data)

            # KL(old||new)
            mu, std = self.pi.forward(obs)
            mu_old, std_old = self.pi_old.forward(obs)
            dist = torch.distributions.Normal(mu, std)
            dist_old = torch.distributions.Normal(mu_old, std_old)
            kl = torch.distributions.kl_divergence(dist_old, dist).sum(-1).mean()

            set_params(self.pi, old_flat)
            return surr_r_new.item(), surr_c_new.item(), kl.item()

        old_flat = flat_params(self.pi).detach()
        surr_r_base, surr_c_base, _ = eval_at(old_flat)

        ok = False
        frac = 1.0
        for _ in range(self.cfg.backtrack_iters):
            new_flat = old_flat + frac * step_dir
            surr_r_new, surr_c_new, kl_new = eval_at(new_flat)
            improve = surr_r_new - surr_r_base

            # cost_ok = (surr_c_new + c_res) <= 0.0
            # if kl_new <= self.cfg.max_kl and cost_ok and improve >= 0.0:
            #     set_params(self.pi, new_flat)
            #     ok = True
            #     break
            lin_feas = (torch.dot(b, new_flat - old_flat).item() + c_res) <= 0.0
            if kl_new <= self.cfg.max_kl and lin_feas and improve >= 0.0:
                set_params(self.pi, new_flat)
                ok = True
                break
            frac *= self.cfg.backtrack_ratio

        info.update(dict(ls_ok=ok, ls_frac=(frac if ok else 0.0)))

        # Fit value functions (reward and cost critics)
        for _ in range(self.cfg.vf_iters):
            v_loss = ((self.vf(obs).squeeze(-1) - ret_r) ** 2).mean()
            self.vf_opt.zero_grad(); v_loss.backward(); 
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 10.0)
            self.vf_opt.step()
        for _ in range(self.cfg.cost_vf_iters):
            c_loss = ((self.cvf(obs).squeeze(-1) - ret_c) ** 2).mean()
            self.cvf_opt.zero_grad(); c_loss.backward(); 
            torch.nn.utils.clip_grad_norm_(self.cvf.parameters(), 10.0)
            self.cvf_opt.step()

        return info
