import torch
from gpytorch.distributions import MultivariateNormal
from statin_agent.active.loop import FastOrdinalLikelihood
from statin_agent.active.sdm.likelihood import OrdinalRegressionLikelihood


def test_fast_likelihood_matches():
    th = torch.tensor([-0.5, 0.3, 1.2])
    slow, fast = OrdinalRegressionLikelihood(th), FastOrdinalLikelihood(th)
    fast.load_state_dict(slow.state_dict())
    dist = MultivariateNormal(torch.randn(7), torch.eye(7) * 0.5)
    y = torch.tensor([0, 1, 2, 3, 1, 0, 3.])
    torch.manual_seed(0); a = slow.expected_log_prob(y, dist)
    torch.manual_seed(0); b = fast.expected_log_prob(y, dist)
    assert torch.allclose(a, b, atol=1e-5)
