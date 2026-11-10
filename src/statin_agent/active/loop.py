"""Pool-based active review with the SDM 2026 ODGP model.

One step: fit the ODGP on the patients reviewed so far, score every unreviewed
patient with a selection rule, review the top one (observe its label, pay its
cost), repeat. The model, likelihood and straddle acquisition are imported
unchanged from `sdm/`; the training loop mirrors the SDM notebook. Every place
where this setting differs from the paper is marked ADAPTATION and listed in
docs/active_review_notes.md.
"""
from __future__ import annotations
import numpy as np
import torch
from gpytorch.mlls.variational_elbo import VariationalELBO
from sklearn.metrics import f1_score, balanced_accuracy_score

from .sdm.likelihood import OrdinalRegressionLikelihood
from .sdm.model import DeepVariationalOrdinalGPModel
from .sdm.acquisition import straddle_acquisition

STRATEGIES = ["random", "greedy", "straddle", "cost_aware_straddle"]


class FastOrdinalLikelihood(OrdinalRegressionLikelihood):
    """ADAPTATION 4 (speed only): the same Monte Carlo expected log-likelihood as
    the original class, computed for all samples at once instead of in a Python
    loop over samples. The math is unchanged; see test_fast_likelihood_matches."""

    def expected_log_prob(self, observations, function_dist, num_samples: int = 100):
        f = function_dist.rsample(torch.Size([num_samples]))                 # (S, N)
        inf = torch.tensor([float("inf")], device=f.device, dtype=f.dtype)
        th = torch.cat([-inf, self.thresholds, inf])                         # (K + 1,)
        cdf = self.normal_cdf(th.view(1, 1, -1) - f.unsqueeze(-1))           # (S, N, K + 1)
        probs = torch.clamp(cdf[..., 1:] - cdf[..., :-1], min=1e-6)          # (S, N, K)
        logp = torch.log(probs).gather(-1, observations.long().view(1, -1, 1).expand(f.shape[0], -1, 1))
        return logp.squeeze(-1).mean(0)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def fit(train_x, train_y, thresholds, hidden_dim=25, feature_dim=30, steps=1500, lr=1e-3):
    """Fresh model each step, as in the SDM notebook (Adam, lr 1e-3, 1500 steps)."""
    likelihood = FastOrdinalLikelihood(thresholds=thresholds.clone())
    model = DeepVariationalOrdinalGPModel(train_x, train_x.shape[1], hidden_dim, feature_dim)
    mll = VariationalELBO(likelihood, model, num_data=train_x.size(0))
    opt = torch.optim.Adam(mll.parameters(), lr=lr)
    model.train(); likelihood.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()
    model.eval(); likelihood.eval()
    return model, likelihood


def predict_gap_prob(model, likelihood, x):
    with torch.no_grad():
        return likelihood(model(x))[:, 1]          # P(class 1 = care gap)


def select(strategy, model, likelihood, cand_x, cand_cost, rng, beta, lam):
    if strategy == "random":
        return int(rng.integers(len(cand_x)))
    if strategy == "greedy":                        # exploit: most likely care gap
        return int(torch.argmax(predict_gap_prob(model, likelihood, cand_x)))
    # ADAPTATION 1: two classes -> one threshold, so gamma = thresholds[0]
    # (the notebook used thresholds[1], the boundary between damage levels 1 and 2).
    gamma = likelihood.thresholds.detach()[0]
    acq = straddle_acquisition(model, cand_x, gamma=gamma, beta=beta)
    if strategy == "cost_aware_straddle":
        # ADAPTATION 2: the paper's penalty is the squared travel distance from the
        # UAV's current position (path-dependent). Reviewing a patient has a fixed
        # cost of its own, so the penalty is lambda * cost(x), cost scaled to [0, 1].
        acq = acq - lam * cand_cost
    return int(torch.argmax(acq))


def run(X, y, cost, strategy, n_init=10, n_iter=100, beta=0.2, lam=0.5, steps=1500,
        seed=0, eval_every=5, log=print):
    """X: (N, D) scaled features; y: (N,) 0/1 care gap; cost: (N,) raw cost per review."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    N = len(y)
    X_t, y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    cost_n = torch.tensor(cost / cost.max(), dtype=torch.float32)
    observed = list(rng.choice(N, size=n_init, replace=False))      # same start for every strategy
    thresholds = torch.linspace(-1.0, 1.0, 1)                         # K - 1 = 1 threshold (notebook init)
    rows, model, likelihood = [], None, None
    for it in range(n_iter + 1):
        obs = np.array(observed)
        cand = np.setdiff1d(np.arange(N), obs)
        need_model = strategy != "random" or it % eval_every == 0 or it == n_iter
        if need_model:
            # ADAPTATION 3: the model is trained before the first selection
            # (the notebook selected the first point with an untrained model).
            model, likelihood = fit(X_t[obs], y_t[obs], thresholds, steps=steps)
            thresholds = likelihood.thresholds.detach()
        if it % eval_every == 0 or it == n_iter:
            p = predict_gap_prob(model, likelihood, X_t[cand]).numpy()
            pred = (p >= 0.5).astype(int)
            rows.append({"strategy": strategy, "seed": seed, "step": it, "n_reviewed": len(obs),
                         "cost_spent": float(cost[obs].sum()), "gaps_found": int(y[obs].sum()),
                         "gaps_total": int(y.sum()),
                         # found by review + flagged correctly by the model among the unreviewed
                         "gaps_identified": int(y[obs].sum() + ((pred == 1) & (y[cand] == 1)).sum()),
                         "false_alarms_unreviewed": int(((pred == 1) & (y[cand] == 0)).sum()),
                         "f1_gap_unreviewed": f1_score(y[cand], pred, zero_division=0),
                         "bal_acc_unreviewed": balanced_accuracy_score(y[cand], pred)})
            log(f"  {strategy:<20} seed {seed} step {it:>3}: gaps found {rows[-1]['gaps_found']}/{y.sum()} "
                f"cost {rows[-1]['cost_spent']:.3g} F1(unreviewed) {rows[-1]['f1_gap_unreviewed']:.2f}")
        if it == n_iter or not len(cand):
            break
        k = select(strategy, model, likelihood, X_t[cand], cost_n[cand], rng, beta, lam)
        observed.append(int(cand[k]))
    return rows
