# Copied unchanged from Boyang Xu's SDM 2026 code (adaptive sampling for post-disaster damage assessment).
import numpy as np
from scipy.stats import norm
import torch
from torch.distributions import Normal
    
def ei(test_x, model, train_y, xi=0.01):
    """
    Compute the Expected Improvement (EI) acquisition function.

    Args:
        test_x (torch.Tensor): Points at which to evaluate EI, shape (N, D).
        model: Trained GP model.
        train_y (torch.Tensor): Observed values for the training data.
        xi (float): Exploration-exploitation trade-off parameter.

    Returns:
        torch.Tensor: EI values for the test points, shape (N,).
    """
    with torch.no_grad():
        posterior = model(test_x)  # Posterior over the test points
        mean_v = posterior.mean  # Posterior mean
        std = posterior.variance.sqrt()
        best_value = train_y.max()  # Best observed value

        # Compute improvement
        improvement = mean_v - best_value - xi
        improvement = torch.clamp(improvement, min = 0)
        # print("*******************improvement", improvement.unique())
        std = std + 1e-9  # Add small value for numerical stability
        Z = improvement / std

        # Compute Expected Improvement
        ei_value = improvement * norm.cdf(Z.cpu().numpy()) + std * norm.pdf(Z.cpu().numpy())
        # print("*******************ei_value", ei_value.unique())
        return torch.tensor(ei_value, dtype=torch.float32)

def straddle_acquisition(model, test_x, gamma, beta=1.0):
    # Evaluate the posterior
    model.eval()
    with torch.no_grad():
        posterior = model(test_x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        # Define the standard normal distribution
        normal = Normal(0,1)

        # compute z scores
        z = (gamma - mean) / std

        # compute pi using Pytorch's cdf
        pi = normal.cdf(z)

        # Compute the straddle function
        straddle = -torch.abs(pi - 0.5) + beta * std
    
    return straddle

def modified_straddle_acquisition(model, test_x, gamma, x_prev, beta=1.0, lambda_penalty=0.1):
    """
    Modified straddle acquisition function that includes a traveling cost penalty.
    
    Args:
        model: The GP model.
        test_x (Tensor): Candidate input(s), shape (N, D).
        gamma (float or Tensor): A threshold parameter used in computing z-scores.
        x_prev (Tensor): The previous/current state, shape (1, D) or (N, D) if using batch.
        beta (float): Weighting factor for the uncertainty term.
        lambda_penalty (float): Weighting factor for the traveling cost penalty.
        
    Returns:
        Tensor: Acquisition value for each test point.
    """
    model.eval()
    with torch.no_grad():
        # Evaluate the GP posterior at test_x
        posterior = model(test_x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        # Standard normal distribution for computing CDF
        normal = Normal(0, 1)

        # Compute z-scores and then pi
        z = (gamma - mean) / std
        pi = normal.cdf(z)

        # Compute the original straddle acquisition function
        straddle = -torch.abs(pi - 0.5) + beta * std

        # Compute the traveling cost as a penalty
        # Here we use the squared Euclidean distance between test_x and x_prev.
        # If x_prev is a single state (1, D), it'll be broadcasted to all test_x.
        # traveling_cost = lambda_penalty * torch.sum((test_x - x_prev)**2, dim=-1)

        # Compute the traveling cost as a penalty using only the first 2 columns (longitude and latitude)
        traveling_cost = lambda_penalty * torch.sum((test_x[:, :2] - x_prev[:, :2])**2, dim=-1)
        
        # Subtract the traveling cost from the straddle value.
        acquisition_value = straddle - traveling_cost

    return acquisition_value

# add random noise acquisition function 

def misclassification_acquisition(model, test_x, gamma):
    """
    Compute the misclassification probability acquisition function at a given point x.

    The misclassification function is defined as:
    
        α_misclass(x) = min(π(x|Dn), 1 - π(x|Dn))

    where:
    - π(x|Dn) is the predicted probability that f(x) lies inside the target level set.
      This is essentially the posterior probability that f(x) ≤ γ.
    
    Intuition:
    The function is highest when π(x|Dn) = 0.5, indicating maximum uncertainty 
    about whether x belongs inside or outside the sub-level set. This encourages 
    sampling points where the model is most unsure, helping to refine the boundary.

    Parameters
    ----------
    pi_x : float or np.ndarray
        The predicted probability π(x|Dn). Expected to be in the interval [0,1].

    Returns
    -------
    float or np.ndarray
        The value of the misclassification probability acquisition function at x.
    """
    model.eval()
    with torch.no_grad():
        posterior = model(test_x)
        mean = posterior.mean
        std = posterior.variance.sqrt()
    
    # Define the standard normal distribution
    normal = Normal(0,1)

    # compute z scores
    z = (gamma - mean) / std

    # compute pi using Pytorch's cdf
    pi = normal.cdf(z)

    return np.minimum(pi, 1 - pi)

def entropy_acquisition(model, test_x, gamma):
    """
    Compute the binary entropy acquisition function at a given point x.

    The binary entropy function for a probability p is defined as:
    
        H_b(p) = -p * log2(p) - (1 - p) * log2(1 - p)
        
    In our case:
    
        α_entropy(x) = H_b(π(x|Dn))
                     = -π(x|Dn)*log2(π(x|Dn)) - (1 - π(x|Dn))*log2(1 - π(x|Dn))
    
    Where π(x|Dn) is the probability that f(x) lies inside the target level set.

    Parameters
    ----------
    pi_x : float or np.ndarray
        The predicted probability π(x|Dn). Should be in the interval [0,1].

    Returns
    -------
    float or np.ndarray
        The value of the binary entropy at x.

    Notes
    -----
    - Entropy is highest at π(x|Dn) = 0.5.
    - To avoid issues with log2(0), we clip π(x|Dn) values to (0,1).

    Raises
    ------
    ValueError
        If pi_x is outside the range [0, 1].
    """
    model.eval()
    with torch.no_grad():
        posterior = model(test_x)
        mean = posterior.mean
        std = posterior.variance.sqrt()
    
    # Define the standard normal distribution
    normal = Normal(0,1)

    # compute z scores
    z = (gamma - mean) / std

    # compute pi using Pytorch's cdf
    pi = normal.cdf(z)


    return -(pi) * np.log2(pi) - (1 - pi) * np.log2(1 - pi)


    