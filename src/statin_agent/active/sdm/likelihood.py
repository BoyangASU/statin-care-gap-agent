# Copied unchanged from Boyang Xu's SDM 2026 code (adaptive sampling for post-disaster damage assessment).
import torch
from torch import Tensor
from torch.distributions.normal import Normal
import torch.nn as nn
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import MultivariateNormal
from typing import Any
import torch.nn.functional as F
import gpytorch
from gpytorch import Module

class OrdinalRegressionLikelihood(_OneDimensionalLikelihood):
    def __init__(self, thresholds: Tensor):
        """
        Initialize the ordinal probit likelihood.

        Args:
            thresholds (Tensor): A tensor of thresholds (θ values) of size (K-1).
                                 Must be sorted in ascending order.
        """
        super().__init__()
        diffs = thresholds.diff()
        self.raw_diffs = torch.nn.Parameter(torch.log(diffs))
        self.base = torch.nn.Parameter(thresholds[0].clone())
        self.normal_cdf = Normal(0, 1).cdf  # Standard normal CDF
        self.sigma = torch.nn.Parameter(torch.tensor(1.0))
    
    @property
    def thresholds(self) -> Tensor:
        return torch.cat([
            self.base.unsqueeze(0),
            self.base + torch.cumsum(torch.exp(self.raw_diffs), dim=0)
        ])

    def forward(self, function_samples: Tensor) -> Tensor:
        """
        Compute probabilities for all classes given latent function samples.

        Args:
            function_samples (Tensor): Samples from the GP latent function of shape (N,).
            
        Returns:
            Tensor: Class probabilities of shape (N, K), where K is the number of classes.
        """ 
        device = function_samples.device
        dtype = function_samples.dtype
        thresholds = torch.cat([
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            self.thresholds,
            torch.tensor([float('inf')], device=device, dtype=dtype)
        ])
        
        class_probs = []
        for k in range(1, len(thresholds)):
            lower_bound = thresholds[k - 1] - function_samples
            upper_bound = thresholds[k] - function_samples
            prob = (self.normal_cdf(upper_bound) - self.normal_cdf(lower_bound)).clamp(min=1e-6)
            class_probs.append(prob)
            
        class_probs = torch.stack(class_probs, dim=-1)  # Shape: (N, K)
        class_probs = class_probs.mean(dim=0)
        return class_probs  # Shape: (K, )

    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, num_samples: int = 100) -> Tensor:
        """
        Compute the expected log-likelihood by integrating over the latent function via Monte Carlo sampling.
        
        Args:
            observations (Tensor): Observed class labels of shape (N,).
            function_dist (MultivariateNormal): The GP's latent function distribution.
            num_samples (int): Number of samples to use for Monte Carlo integration.
        
        Returns:
            Tensor: The estimated expected log-likelihood of shape (N,).
        """
        # Sample from the latent function distribution.
        # f_samples shape: (num_samples, N)
        f_samples = function_dist.rsample(torch.Size([num_samples]))
        
        device = function_dist.mean.device
        dtype = function_dist.mean.dtype
        thresholds = torch.cat([
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            self.thresholds,
            torch.tensor([float('inf')], device=device, dtype=dtype)
        ])
        
        # Compute log likelihood for each sample.
        log_probs = []
        for i in range(num_samples):
            f_sample = f_samples[i]  # shape: (N,)
            log_prob_sample = torch.zeros_like(f_sample)
            for k in range(1, len(thresholds)):
                lower_bound = thresholds[k - 1] - f_sample
                upper_bound = thresholds[k] - f_sample
                # Compute the probability for the interval.
                prob = self.normal_cdf(upper_bound) - self.normal_cdf(lower_bound)
                log_prob_sample += (observations == (k - 1)).float() * torch.log(torch.clamp(prob, min=1e-6))
            log_probs.append(log_prob_sample)
        # Average over samples
        log_probs = torch.stack(log_probs, dim=0)
        expected_log_prob = log_probs.mean(dim=0)
        return expected_log_prob
