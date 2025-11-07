# safe_fqi.py
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- small utils -----------------------
# def mlp(sizes, act=nn.ReLU, out_act=None):
#     layers = []
#     for i in range(len(sizes)-1):
#         layers += [nn.Linear(sizes[i], sizes[i+1])]
#         if i < len(sizes)-2:
#             layers += [act()]
#         elif out_act is not None:
#             layers += [out_act()]
#     return nn.Sequential(*layers)



# class QNet(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden=(64,64)):
#         super().__init__()
#         self.net = mlp([obs_dim+act_dim, *hidden, 1])
#     def forward(self, obs, act):
#         x = torch.cat([obs, act], dim=-1)
#         return self.net(x)  # (B,1)

class RunningNorm(nn.Module):
    """Online standardization for inputs; avoids huge gradients."""
    def __init__(self, dim, eps=1e-6, mom=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var",  torch.ones(dim))
        self.mom = mom; self.eps = eps
        self._initted = False
    @torch.no_grad()
    def update(self, x):
        m = x.mean(dim=0); v = x.var(dim=0, unbiased=False)
        if not self._initted:
            self.mean.copy_(m); self.var.copy_(v + self.eps); self._initted = True
        else:
            self.mean.lerp_(m, self.mom); self.var.lerp_(v + self.eps, self.mom)
    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var)

def mlp_tanh(sizes):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [nn.Tanh()]
    return nn.Sequential(*layers)

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.act_norm = RunningNorm(act_dim)
        self.body = mlp_tanh([obs_dim+act_dim, *hidden, 1])
    @torch.no_grad()
    def update_norms(self, obs, act):
        self.obs_norm.update(obs)
        self.act_norm.update(act)
    def forward(self, obs, act):
        x = torch.cat([self.obs_norm(obs), self.act_norm(act)], dim=-1)
        return self.body(x)  # (B,1)




class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6), device="cpu"):
        self.obs_buf  = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf  = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rew_buf  = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.cost_buf = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.max_size, self.ptr, self.size = size, 0, 0
        self.device = device

    def store(self, *args, **kwargs):
        """Insert one or many transitions into the replay buffer.

        Accepts either positional arguments (o, a, r, c, o2, d) for a single
        transition or numpy/torch batches, or a single dict with keys
        {"obs","act","rew","cost","next_obs","done"} containing
        equal-length arrays for batch insertion.
        """
        # Normalize inputs to tensors on the correct device
        if len(args) == 1 and isinstance(args[0], dict):
            data = args[0]
            o  = data.get("obs")
            a  = data.get("act")
            r  = data.get("rew")
            c  = data.get("cost")
            o2 = data.get("next_obs")
            d  = data.get("done")
        else:
            if len(args) >= 6:
                o, a, r, c, o2, d = args[:6]
            else:
                o = kwargs.get("o")
                a = kwargs.get("a")
                r = kwargs.get("r")
                c = kwargs.get("c")
                o2 = kwargs.get("o2")
                d = kwargs.get("d")

        o_t  = torch.as_tensor(o,  device=self.device, dtype=torch.float32)
        a_t  = torch.as_tensor(a,  device=self.device, dtype=torch.float32)
        r_t  = torch.as_tensor(r,  device=self.device, dtype=torch.float32).view(-1, 1)
        c_t  = torch.as_tensor(c,  device=self.device, dtype=torch.float32).view(-1, 1)
        o2_t = torch.as_tensor(o2, device=self.device, dtype=torch.float32)
        d_t  = torch.as_tensor(d,  device=self.device, dtype=torch.float32).view(-1, 1)

        # Ensure 2D for observations/actions
        if o_t.dim() == 1:
            o_t = o_t.view(1, -1)
        if o2_t.dim() == 1:
            o2_t = o2_t.view(1, -1)
        if a_t.dim() == 1:
            a_t = a_t.view(1, -1)

        n = o_t.shape[0]
        assert a_t.shape[0] == n and o2_t.shape[0] == n and r_t.shape[0] == n and c_t.shape[0] == n and d_t.shape[0] == n, "Mismatched batch lengths in buffer.store()"

        # Fast path for single transition
        if n == 1:
            self.obs_buf[self.ptr]  = o_t[0]
            self.act_buf[self.ptr]  = a_t[0]
            self.rew_buf[self.ptr]  = r_t[0]
            self.cost_buf[self.ptr] = c_t[0]
            self.obs2_buf[self.ptr] = o2_t[0]
            self.done_buf[self.ptr] = d_t[0]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            return

        # Vectorized circular insertion for batches
        start = self.ptr
        end = self.ptr + n
        if end <= self.max_size:
            self.obs_buf[start:end]  = o_t
            self.act_buf[start:end]  = a_t
            self.rew_buf[start:end]  = r_t
            self.cost_buf[start:end] = c_t
            self.obs2_buf[start:end] = o2_t
            self.done_buf[start:end] = d_t
        else:
            first = self.max_size - start
            second = n - first
            # First slice to end of buffer
            self.obs_buf[start:self.max_size]  = o_t[:first]
            self.act_buf[start:self.max_size]  = a_t[:first]
            self.rew_buf[start:self.max_size]  = r_t[:first]
            self.cost_buf[start:self.max_size] = c_t[:first]
            self.obs2_buf[start:self.max_size] = o2_t[:first]
            self.done_buf[start:self.max_size] = d_t[:first]
            # Wrap-around slice from beginning
            self.obs_buf[0:second]  = o_t[first:]
            self.act_buf[0:second]  = a_t[first:]
            self.rew_buf[0:second]  = r_t[first:]
            self.cost_buf[0:second] = c_t[first:]
            self.obs2_buf[0:second] = o2_t[first:]
            self.done_buf[0:second] = d_t[first:]

        self.ptr = end % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(
            obs  = self.obs_buf[idxs],
            act  = self.act_buf[idxs],
            rew  = self.rew_buf[idxs],
            cost = self.cost_buf[idxs],
            obs2 = self.obs2_buf[idxs],
            done = self.done_buf[idxs],
        )

# ----------------------- Q and Cost models -----------------------

class CostHead(nn.Module):
    """Predicts mean and log-variance; trained via Gaussian NLL."""
    def __init__(self, obs_dim, act_dim, hidden=(64,64)):
        super().__init__()
        self.backbone = mlp_tanh([obs_dim+act_dim, *hidden, hidden[-1]])
        self.mu = nn.Linear(hidden[-1], 1)
        self.log_var = nn.Linear(hidden[-1], 1)
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        h = self.backbone(x)
        mu = torch.sigmoid(self.mu(h))       # clamp into [0,1]
        log_var = torch.clamp(self.log_var(h), min=-10.0, max=5.0)
        return mu, log_var

class CostEnsemble(nn.Module):
    """Simple epistemic+aleatoric UQ via ensemble."""
    def __init__(self, obs_dim, act_dim, num=5, hidden=(256,256), device="cpu"):
        super().__init__()
        self.members = nn.ModuleList([CostHead(obs_dim, act_dim, hidden) for _ in range(num)])
        self.device = device
        self.num = num

    def forward_all(self, obs, act):
        mus, vars_ = [], []
        for m in self.members:
            mu, logv = m(obs, act)
            mus.append(mu)
            vars_.append(torch.exp(logv))
        mu = torch.stack(mus, dim=0)     # (E,B,1)
        var = torch.stack(vars_, dim=0)  # (E,B,1)
        return mu, var

    @torch.no_grad()
    def predict(self, obs, act):
        mu_e, var_e = self.forward_all(obs, act)
        mu_bar = mu_e.mean(dim=0)                        # (B,1)
        # total variance = aleatoric mean + epistemic variance of means
        alea = var_e.mean(dim=0)                         # (B,1)
        epi  = mu_e.var(dim=0, unbiased=False)           # (B,1)
        total_var = alea + epi
        std = torch.sqrt(torch.clamp(total_var, 1e-12))
        return mu_bar, std

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha_pos=0.2):
        super().__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
    def forward(self, logits, targets):
        # targets in {0,1}
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = p*targets + (1-p)*(1-targets)
        alpha = self.alpha_pos*targets + (1-self.alpha_pos)*(1-targets)
        loss = alpha * (1-pt).pow(self.gamma) * ce
        return loss.mean()

class CostClassifierEnsemble(nn.Module):
    """
    Ensemble classifier for binary hazard c∈{0,1}.
    Trains with BCEWithLogitsLoss (+pos_weight for class imbalance).
    predict() returns (p_mean, p_std) with total uncertainty (epistemic+aleatoric).
    """
    def __init__(self, obs_dim, act_dim, M=5, hidden=(64,64), device="cpu", pos_weight=10.0):
        super().__init__()
        self.members = nn.ModuleList([
            nn.Sequential(
                mlp_tanh([obs_dim + act_dim, *hidden, hidden[-1]]),
                nn.Linear(hidden[-1], 1)  # logits
            ) for _ in range(M)
        ])
        self.device = device
        self.M = M
        # BCE with positive class upweighting (tune pos_weight to your hazard rate)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        self.criterion = FocalBCEWithLogitsLoss(gamma=1.5, alpha_pos=0.6)
    def _forward_all_logits(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        logits = []
        for m in self.members:
            logits.append(m(x).squeeze(-1))     # [B]
        return torch.stack(logits, dim=0)        # [M,B]

    @torch.no_grad()
    def predict(self, obs, act):
        """
        Returns:
          p_mean: [B,1]   ensemble-mean hazard prob
          p_std:  [B,1]   sqrt( var_epistemic + var_aleatoric_mean )
        """
        logits = self._forward_all_logits(obs, act)      # [M,B]
        P = torch.sigmoid(logits)                        # [M,B]
        p_mean = P.mean(dim=0, keepdim=True).T           # [B,1]
        var_ep = P.var(dim=0, unbiased=False, keepdim=True).T         # [B,1]
        var_al = (P * (1 - P)).mean(dim=0, keepdim=True).T            # [B,1]
        p_std = (var_ep + var_al).sqrt()
        return p_mean.clamp(0, 1), p_std
    
# ----------------------- Safe FQI Agent -----------------------
class SafeFQI:
    def __init__(
        self, obs_dim, act_dim, device="cpu",
        gamma=0.99, lr_q=2e-4, lr_cost=3e-4,
        hidden=(64,64),
        # action sampling
        act_low=-1.0, act_high=1.0, action_candidates=512, target_candidates= 512 ,
        # safety gate
        beta0=0.2, beta1=1.1, safe_threshold=0.8 , thresh_final=0.2, anneal_steps=400_000,penalty_clip = (2,30.0),
        # training
        tau=0.03, batch_size=256,
        ensemble_size=5,
        penalty=1
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.low = act_low; self.high = act_high
        self.K = action_candidates
        self.M = target_candidates

        self.q = QNet(obs_dim, act_dim, hidden).to(device)
        self.q_targ = QNet(obs_dim, act_dim, hidden).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr_q, weight_decay=1e-5)
        
        self.q2 = QNet(obs_dim, act_dim, hidden).to(device)
        self.q2_targ = QNet(obs_dim, act_dim, hidden).to(device)
        self.q2_targ.load_state_dict(self.q2.state_dict())
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr_q, weight_decay=1e-5)



        self.cost = CostClassifierEnsemble(obs_dim, act_dim, M=ensemble_size, hidden=hidden, device=device, pos_weight=10.0).to(device)
        # self.cost = CostEnsemble(obs_dim, act_dim, num=ensemble_size, hidden=hidden, device=device).to(device)
        self.cost_optims = [torch.optim.Adam(h.parameters(), lr=lr_cost) for h in self.cost.members]

        self.tau = tau
        self.global_step = 0
        self.penalty=penalty      
        self.beta0 =beta0; self.beta1 = beta1
        self.thresh0 = safe_threshold
        self.thresh1 = thresh_final
        self.anneal_steps = max(1, anneal_steps)
        self.penalty=penalty
        self.eta_lr = 0.05          # dual step size; tune 0.01–0.2
        self.penalty_clip = (0,12.0)
        self.buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6), device=device)
        self._cost_rate_ema = 0.0
        self._ema_alpha = 0.05
        self.warmup_steps = 10000        
        self.reward_scale = 1500.0  # start here given r≈1e-3; tune 300–3000
        self.cost_scale   = 1.0     # usually keep cost as 0/1
   

        self.rew_ema_mean, self.rew_ema_std, self.rew_ema_beta = 0.0, 1.0, 0.01
        self.target_rew_std = 10.0


    def _scale_reward(self, r):
        # r is (B,1) or (B,)
        with torch.no_grad():
            m = float(r.mean())
            s = float(r.std().clamp_min(1e-6))
            self.rew_ema_mean = (1-self.rew_ema_beta)*self.rew_ema_mean + self.rew_ema_beta*m
            self.rew_ema_std  = (1-self.rew_ema_beta)*self.rew_ema_std  + self.rew_ema_beta*s
        r_norm = (r - self.rew_ema_mean) / max(1e-6, self.rew_ema_std)
        return self.target_rew_std * r_norm

        # return self.reward_scale * r

    def _anneal(self):
        # Progress in [0,1]
        t = min(1.0, self.global_step / self.anneal_steps)
        # Interpolate from (beta0, thresh0) → (beta1, thresh1)
        beta   = (1 - t) * self.beta0   + t * self.beta1
        thresh = (1 - t) * self.thresh0 + t * self.thresh1
        return float(beta), float(thresh)

    @torch.no_grad()
    def _sample_actions(self, n, k):
        # uniform in [-1,1]
        a = torch.empty((n, k, self.act_dim), device=self.device).uniform_(self.low, self.high)
        return a

    @torch.no_grad()
    def _cem_safe(self, obs, iters=3, pop=64, elites=32):
        mean = torch.zeros(self.act_dim, device=self.device)
        std  = torch.ones(self.act_dim, device=self.device)
        beta, thresh = self._anneal()
        self.sel_penalty = 2
        for _ in range(iters):
            A = (mean + std * torch.randn(pop, self.act_dim, device=self.device)).clamp(self.low, self.high)
            O = obs.expand(pop, -1)
            if self.global_step < self.warmup_steps:
                ucb = torch.zeros(pop, device=self.device)
            else:
                mu, sd = self.cost.predict(O, A)
                ucb = (mu + beta*sd).squeeze(-1)
            # score: prefer safe; if none safe, penalize by UCB
            q1 = self.q(O, A).squeeze(-1)
            q2 = self.q2(O, A).squeeze(-1)
            q  = torch.minimum(q1, q2)
            score = torch.where(ucb <= thresh, q, q - self.sel_penalty * ucb)
            idx = torch.topk(score, k=elites, dim=0).indices
            elite = A[idx]
            mean = elite.mean(dim=0); std = elite.std(dim=0).clamp_min(0.05)
        # final pick among last pop
        return A[idx[0]]
        
    @torch.no_grad()
    def _q_max_over_samples(self, obs, k):
        # obs: (B,D) -> sample k actions per obs, choose max Q
        B = obs.shape[0]
        a = torch.empty((B, k, self.act_dim), device=self.device).uniform_(self.low, self.high)
        obs_rep = obs.unsqueeze(1).expand(B, k, obs.shape[-1]).reshape(B*k, -1)
        a_flat = a.reshape(B*k, -1)
        q1 = self.q_targ(obs_rep, a_flat).view(B, k, 1)
        q2 = self.q2_targ(obs_rep, a_flat).view(B, k, 1)
        q = torch.minimum(q1, q2)
        qmax, _ = q.max(dim=1)   # (B,1)
        return qmax


        

    @torch.no_grad()
    def _q_max_over_samples_safe(self, obs, k: int):
        """
        Sample k actions at s' and compute a *safe* target by:
        1) masking actions whose UCB cost > threshold,
        2) taking max-Q among safe actions when available,
        3) otherwise falling back to maximizing (Q - penalty * UCB).
        Returns shape (B,1).
        """
        B = obs.shape[0]

        # Sample k actions uniformly in the action box
        a = torch.empty((B, k, self.act_dim), device=self.device).uniform_(self.low, self.high)

        # Flatten for batched model calls
        obs_rep = obs.unsqueeze(1).expand(B, k, obs.shape[-1]).reshape(B * k, -1)
        a_flat  = a.reshape(B * k, -1)

        # Predict cost mean/std and form UCB
        mu, std = self.cost.predict(obs_rep, a_flat)
        # Robustness: std >= 0
        std = torch.clamp(std, min=1e-8)

        beta, thresh = self._anneal()               # floats
        ucb = (mu + beta * std).view(B, k)          # (B,k)

        # Mask out unsafe candidates
        safe_mask = (ucb <= thresh)                 # (B,k)  <-- BOOL
        q1 = self.q_targ(obs_rep, a_flat).view(B, k)
        q2 = self.q2_targ(obs_rep, a_flat).view(B, k)
        q = torch.min(q1, q2)  # use min for targets

        # Max over safe actions (fill unsafe with -inf)
        neg_inf = torch.full_like(q, -1e9)
        q_safe = torch.where(safe_mask, q, neg_inf)        # (B,k)
        qmax_safe, _ = q_safe.max(dim=1)               # (B,)

        # Fallback: maximize (Q - penalty * UCB) when no safe actions exist
        # qmax_fallback, _ = (q - float(self.penalty) * ucb).max(dim=1)      # (B,)
        qmax_fallback = (q - self.penalty * torch.clamp(ucb, max=1.0)).max(dim=1).values

        # Did we have any safe actions per state?
        has_safe = safe_mask.any(dim=1)                    # (B,) BOOL

        # Choose safe max when available; else fallback
        qmax = torch.where(has_safe, qmax_safe, qmax_fallback)  # (B,)
        return qmax.unsqueeze(-1)                          # (B,1)



    @torch.no_grad()
    def _cem_reward_only(self, obs, iters=3, pop=128, elites=64):
        use_targ = (self.global_step <2_000)
        scorer = (lambda O,A: torch.minimum(self.q2_targ(O,A), self.q_targ(O,A))) \
                if use_targ else (lambda O,A: torch.minimum(self.q2(O,A), self.q(O,A)))
        mean = torch.zeros(self.act_dim, device=self.device)
        std  = torch.ones(self.act_dim, device=self.device)
        for _ in range(iters):
            A = (mean + std * torch.randn(pop, self.act_dim, device=self.device)).clamp(self.low, self.high)
            O = obs.expand(pop, -1)
            q  = scorer(O, A).squeeze(-1)
            idx = torch.topk(q, k=elites, dim=0).indices
            elite = A[idx]
            mean = elite.mean(dim=0); std = elite.std(dim=0).clamp_min(0.05)
        return A[idx[0]]

    @torch.no_grad()
    def _q_targ_max_cem(self, obs, iters=3, pop=256, elites=64, noise=0.0):
        B = obs.shape[0]
        mean = torch.zeros(B, self.act_dim, device=self.device)
        std  = torch.ones(B, self.act_dim, device=self.device)
        for _ in range(iters):
            A = (mean.unsqueeze(1) + std.unsqueeze(1) * 
                torch.randn(B, pop, self.act_dim, device=self.device))
            A = A.clamp(self.low, self.high)
            if noise > 0:
                A = (A + noise*torch.randn_like(A)).clamp(self.low, self.high)
            O = obs.unsqueeze(1).expand(B, pop, obs.shape[-1]).reshape(B*pop, -1)
            Aflat = A.reshape(B*pop, -1)

            q1 = self.q2_targ(O, Aflat).view(B, pop, 1)
            q2 = self.q_targ (O, Aflat).view(B, pop, 1)
            q  = torch.minimum(q1, q2).squeeze(-1)
            idx = torch.topk(q, k=elites, dim=1).indices
            elite = torch.gather(A, 1, idx.unsqueeze(-1).expand(B, elites, self.act_dim))
            mean  = elite.mean(dim=1)
            std   = elite.std(dim=1).clamp_min(0.05)
        # final:
        Oe = obs.repeat_interleave(elites, 0)
        Ae = elite.reshape(B*elites, -1)
        qf = torch.minimum(self.q2_targ(Oe, Ae), self.q_targ(Oe, Ae)).view(B, elites, 1).squeeze(-1)
        return qf.max(dim=1).values.unsqueeze(-1)

    # def _q_targ_max_cem_safe(self, obs, iters=5, pop=256, elites=64, noise=0.05, lam_targ=0.5):
    #     # lam_targ << training penalty; only nudges the argmax
    #     mean = torch.zeros(self.act_dim, device=self.device)
    #     std  = torch.ones(self.act_dim, device=self.device)
    #     lam_targ  = 1
    #     beta, thresh = self._anneal()
    #     for _ in range(iters):
    #         A = (mean + std * torch.randn(pop, self.act_dim, device=self.device)).clamp(self.low, self.high)
    #         O = obs.expand(pop, -1)
    #         q = self.q_targ(O, A).squeeze(-1)  # use target net
    #         if self.global_step < self.warmup_steps:
    #             ucb = torch.zeros_like(q)
    #         else:
    #             mu, sd = self.cost.predict(O, A)      # ensemble cost model
    #             ucb = (mu + beta * sd).squeeze(-1)
    #         score = q - lam_targ * torch.relu(ucb - thresh)  # only penalize over-threshold
    #         idx = torch.topk(score, elites, dim=0).indices
    #         elite = A[idx]
    #         mean, std = elite.mean(dim=0), elite.std(dim=0) + noise
    #     # Return the Q value at the found argmax a*
    #     a_star = mean.clamp(self.low, self.high).unsqueeze(0).expand(obs.shape[0], -1)
    #     # Ensure (B,1) shape to match other targets and predictions
    #     return self.q_targ(obs, a_star)

    def _q_targ_max_cem_safe(self, obs, iters=5, pop=256, elites=64, noise=0.05, lam_targ=1.0):
        """Batch-aware safe CEM over target critics.

        Args:
            obs: (B, obs_dim) tensor

        Returns:
            q_targ_max: (B,1) tensor of max target-Q for each state under a safety-aware CEM proposal

        Notes:
            - Safety only steers the *argmax* via score = q - lam_targ*relu(ucb - thresh).
            The returned target is the *raw* Q at the selected action (no penalty leak).
        """
        B = obs.shape[0]
        mean = torch.zeros(B, self.act_dim, device=self.device)
        std  = torch.ones(B, self.act_dim, device=self.device)

        beta, thresh = self._anneal()
        for _ in range(iters):
            # Sample (B, pop, act_dim) from per-state Gaussians
            A = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, pop, self.act_dim, device=self.device)
            A = A.clamp(self.low, self.high)
            if noise > 0:
                A = (A + noise * torch.randn_like(A)).clamp(self.low, self.high)

            # Flatten for networks: (B*pop, ·)
            O = obs.unsqueeze(1).expand(B, pop, obs.shape[-1]).reshape(B*pop, -1)
            Aflat = A.reshape(B*pop, -1)

            # Target critics (twin) and min for stability
            q1 = self.q2_targ(O, Aflat).view(B, pop, 1)
            q2 = self.q_targ (O, Aflat).view(B, pop, 1)
            q  = torch.minimum(q1, q2).squeeze(-1)   # (B, pop)

            # Safety UCB from cost ensemble (batch-aware)
            if self.global_step < self.warmup_steps:
                ucb = torch.zeros_like(q)
            else:
                mu, sd = self.cost.predict(O, Aflat)  # (B*pop,1) each
                ucb = (mu + beta * sd).view(B, pop)   # (B, pop)

            # Score steers which actions are elite; penalty only guides argmax
            score = q - lam_targ * torch.relu(ucb - thresh)

            # Per-state elites
            idx = torch.topk(score, k=elites, dim=1).indices                    # (B, elites)
            elite = torch.gather(A, 1, idx.unsqueeze(-1).expand(B, elites, self.act_dim))
            mean  = elite.mean(dim=1)
            std   = elite.std(dim=1).clamp_min(0.05)

        # Final target value = best *raw* Q among the final elites (no penalty leak)
        Oe = obs.repeat_interleave(elites, 0)                  # (B*elites, obs_dim)
        Ae = elite.reshape(B*elites, -1)                       # (B*elites, act_dim)
        qf = torch.minimum(self.q2_targ(Oe, Ae), self.q_targ(Oe, Ae)).view(B, elites, 1).squeeze(-1)
        return qf.max(dim=1).values.unsqueeze(-1)              # (B,1)


    @torch.no_grad()
    def select_action_safe(self, obs_np):
        self.global_step += 1
        beta, thresh = self._anneal()

        obs = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).view(1,-1)
        # # sample K candidate actions

        # a_np = self._cem_safe(obs).cpu().numpy()
        # return a_np
        if self.global_step < self.warmup_steps:
    # reward-only selection

            a = self._cem_reward_only(obs)
            return a.detach().cpu().numpy()
        else:
            a = self._cem_safe(obs)
            return a.detach().cpu().numpy()



        # a = self._sample_actions(n=1, k=self.K).squeeze(0)  # (K,act_dim)
        # mu, std = self.cost.predict(obs.repeat(self.K,1), a)
        # ucb = mu + beta * std
    
        # safe_mask = (ucb.squeeze(-1) <= thresh)

        # if safe_mask.any():
        #     a_safe = a[safe_mask]
        #     # choose the safe action with highest Q
        #     q = self.q(obs.repeat(a_safe.shape[0],1), a_safe)
        #     q2=self.q2(obs.repeat(a_safe.shape[0],1), a_safe)
        #     q=torch.min(q, q2)
        #     idx = torch.argmax(q.squeeze(-1)).item()
        #     return a_safe[idx].cpu().numpy()
        # else:
        #     # fallback: pick the least-unsafe action (min UCB), to keep moving
        #     q1 = self.q(obs.repeat(self.K,1), a).squeeze(-1)
        #     q2 = self.q2(obs.repeat(self.K,1), a).squeeze(-1)
        #     score = torch.minimum(q1, q2) - self.penalty * ucb.squeeze(-1)
        #     idx = torch.argmax(score).item()
        #     return a[idx].cpu().numpy()

        # pure reward only

        # q = self.q(obs.repeat(a.shape[0],1), a)
        # q2=self.q2(obs.repeat(a.shape[0],1), a)
        # q=torch.min(q, q2)
        # idx = torch.argmax(q.squeeze(-1)).item()
        # return a[idx].cpu().numpy()



        # a = self._cem_reward_only(obs)
        # return a.detach().cpu().numpy()


    def store(self, *args, **kwargs):
        self.buffer.store(*args, **kwargs)




    def update(self, data, ep_cost_mean: float, cost_limit: float, avg_ep_len: float, updates=50):
        logs = {}
        self.store(data)
        for _ in range(updates):
            if self.buffer.size < max(1000, self.batch_size):
                break
            batch = self.buffer.sample_batch(self.batch_size)
            o, a, r, c, o2, d = batch["obs"], batch["act"], batch["rew"], batch["cost"], batch["obs2"], batch["done"]
            cost_rate = float(ep_cost_mean / max(1.0, avg_ep_len))
            self._cost_rate_ema = (1 - self._ema_alpha) * self._cost_rate_ema + self._ema_alpha * cost_rate
            gap = self._cost_rate_ema - float(cost_limit)
            self.penalty = float(np.clip(self.penalty + self.eta_lr * gap, *self.penalty_clip))
            logs["penalty"] = self.penalty
            base_penalty = 0
            # --- Q update (FQI with action-sampling to approximate max over a') ---
            with torch.no_grad():
                if self.global_step < self.warmup_steps:
                    q_targ_max = self._q_max_over_samples(o2, k=self.M)  # no safety mask

                    y = self._scale_reward(r) + self.gamma * (1.0 - d) * q_targ_max
                else:
                    # q_targ_max = self._q_max_over_samples_safe(o2, k=self.M)
                    
                    q_targ_max = self._q_targ_max_cem_safe(o2, iters=3, pop=256, elites=64, noise=0.05,lam_targ=2.0)
                    y = self._scale_reward(r) - ((base_penalty+self.penalty) * self.cost_scale) * c + self.gamma * (1.0 - d) * q_targ_max
                # q_targ_max = self._q_max_over_samples_safe(o2, k=self.M)
                # q_targ_max = self._q_max_over_samples(o2, k=self.M)
                # q_targ_max = self._q_targ_max_cem(o2, iters=3, pop=256, elites=64, noise=0.1)
                # y =self._scale_reward(r)-(self.penalty * self.cost_scale) * c + self.gamma * (1.0 - d) * q_targ_max
                # y =self._scale_reward(r) -(self.penalty * self.cost_scale) * c+ self.gamma * (1.0 - d) * q_targ_max
                y = y.clamp(-2000.0, 2000.0)
            self.q.update_norms(o, a)
            self.q2.update_norms(o, a)

            q1_pred = self.q(o, a)
            # q1_loss = F.mse_loss(q1_pred, y)
            self.q_optim.zero_grad(set_to_none=True)
            q1_loss = F.smooth_l1_loss(q1_pred, y) 
            q1_loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q_optim.step()  

            q2_pred = self.q2(o, a)
            # q2_loss = F.mse_loss(q2_pred, y)
            self.q2_optim.zero_grad(set_to_none=True)
            q2_loss = F.smooth_l1_loss(q2_pred, y)
            q2_loss.backward()
            nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
            self.q2_optim.step()


            # --- Cost ensemble update (Gaussian NLL) ---
            # cost_losses = []
            # for m, opt in zip(self.cost.members, self.cost_optims):
            #     mask = torch.rand(o.shape[0], device=o.device) < 0.5
            #     o_m, a_m, c_m = o[mask], a[mask], c[mask]
            #     if o_m.shape[0] < 32:  # ensure enough samples
            #         o_m, a_m, c_m = o, a, c
            #     mu, log_var = m(o_m, a_m)
            #     var = torch.exp(log_var).clamp_min(1e-10)

            #     # positive reweighting
            #     alpha_pos = 2.0
            #     w = torch.where(c_m > 1e-3, torch.full_like(c_m, alpha_pos), torch.ones_like(c_m))

            #     nll = 0.5 * ((c_m - mu)**2 / var + log_var)
            #     loss = (w * nll).mean()

            #     opt.zero_grad(set_to_none=True)
            #     loss.backward()
            #     nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            #     opt.step()
            #     cost_losses.append(loss.item())

            # --- Cost ensemble update (BCE for 0/1 costs) ---
            cost_losses = []
            # Slight label smoothing to stabilize learning on rare hazards (optional)
            label_smooth = 0.01  # set to 0.0 to disable
            for m, opt in zip(self.cost.members, self.cost_optims):
                # stratified-ish subsample to decorrelate members (optional)
                mask = torch.rand(o.shape[0], device=o.device) < 0.5
                o_m, a_m, c_m = o[mask], a[mask], c[mask]
                if o_m.shape[0] < 32:
                    o_m, a_m, c_m = o, a, c

                # prepare inputs/targets
                x = torch.cat([o_m, a_m], dim=-1)            # [B, obs+act]
                # clamp to [0,1] and optionally smooth
                y = c_m.squeeze(-1).clamp(0.0, 1.0)
                if label_smooth > 0:
                    y = (1 - label_smooth) * y + label_smooth * 0.5

                # forward & loss (logits expected)
                logits = m(x).squeeze(-1)                    # [B]
                loss = self.cost.criterion(logits, y)        # BCEWithLogitsLoss with pos_weight inside the ensemble

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                cost_losses.append(loss.item())

            # --- target update ---
            with torch.no_grad():
                for p, p_t in zip(self.q.parameters(), self.q_targ.parameters()):
                    p_t.data.lerp_(p.data, self.tau)
                for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_t.data.lerp_(p.data, self.tau)
            logs = dict(q1_loss=float(q1_loss.item()),
                        q2_loss=float(q2_loss.item()),
                        cost_loss=float(np.mean(cost_losses)),
                        penalty=self.penalty)
        return logs


def collect_epoch_safe_fqi(env, agent, steps_per_epoch, device):
    o, info = env.reset()
    obs_l, act_l, rew_l, cost_l, next_l, done_l = [], [], [], [], [], []
    ep_costs, ep_rewards = [], []
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    eps_start, eps_final, eps_decay = 0.60, 0.03, 400_000 #0.6
    unsafe_steps, num_eps = 0, 0
 
    for t in range(steps_per_epoch):
        # if agent.global_step <=steps_per_epoch*1:
        #     eps = 1
        # else:
        eps = eps_final + (eps_start - eps_final) * np.exp(- (agent.global_step + t) / eps_decay)
            # eps = eps_final + (eps_start - eps_final) * np.exp(- (agent.global_step + t) / eps_decay)
        obs_np = o.astype(np.float32)

        if np.random.rand() < eps:
            # random -> screen by UCB; fallback to safe selector
            a_np = env.action_space.sample()
            agent.global_step += 1
        #     if agent.global_step >= agent.warmup_steps/2:
        # # try up to 32 random safe draws by UCB
        #         tried = 0
        #         picked = False
        #         while tried < 32:
        #             a_try = env.action_space.sample().astype(np.float32)
        #             o_t = torch.as_tensor(obs_np, device=device).unsqueeze(0)
        #             a_t = torch.as_tensor(a_try, device=device).unsqueeze(0)
        #             mu, sd = agent.cost.predict(o_t, a_t)
        #             beta, thresh = agent._anneal()
        #             ucb = float((mu + beta*sd).item())
        #             if ucb <= thresh:
        #                 a_np = a_try; picked = True; break
        #             tried += 1
        #         if not picked:
        #             a_np = agent.select_action_safe(obs_np)  # safe fallback
        else:
            a_np = agent.select_action_safe(obs_np)

        step_out = env.step(a_np)
 
        o2, r, c, terminated, truncated, info = step_out
        # else:
        #     o2, r, terminated, truncated, info = step_out
        #     c = float(info.get('cost', 0.0))

        terminal = bool(terminated)
        trunc = bool(truncated)
        done = terminal or trunc

        obs_l.append(obs_np)
        act_l.append(a_np.astype(np.float32))
        rew_l.append(float(r))
        cost_l.append(float(c))
        next_l.append(o2.astype(np.float32))
        done_l.append(float(done))

        unsafe_steps += int(float(c) > 0.0)
        ep_len += 1
        ep_cost += float(c)
        ep_ret += float(r)
        o = o2

        if done:
            ep_costs.append(ep_cost)
            ep_rewards.append(ep_ret)
            num_eps += 1
            o, info = env.reset()
            ep_ret, ep_len, ep_cost = 0.0, 0, 0.0

    data = {
        "obs": np.stack(obs_l, axis=0),
        "act": np.stack(act_l, axis=0),
        "rew": np.array(rew_l, dtype=np.float32),
        "cost": np.array(cost_l, dtype=np.float32),
        "next_obs": np.stack(next_l, axis=0),
        "done": np.array(done_l, dtype=np.float32),
        "_unsafe_steps": int(unsafe_steps),
        "_episodes": int(max(1, num_eps))
    }

    ep_cost_mean = float(np.mean(ep_costs)) if len(ep_costs) > 0 else 0.0
    avg_ep_len = steps_per_epoch / max(1, num_eps)
    ep_reward_mean = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0

    return data, ep_cost_mean, avg_ep_len, ep_reward_mean