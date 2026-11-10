# Copied unchanged from Boyang Xu's SDM 2026 code (adaptive sampling for post-disaster damage assessment).
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import AbstractVariationalGP
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, dropout_p=0.2, use_batchnorm=True):
        """
        A feature extractor with:
        - Two hidden layers (both of dimension hidden_dim).
        - Optional batch normalization after each hidden layer.
        - ReLU activations.
        - Dropout to reduce overfitting.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of each hidden layer.
            feature_dim (int): Output feature dimension for the GP layer.
            dropout_p (float): Dropout probability. Default is 0.2.
            use_batchnorm (bool): Whether to use batch normalization.
        """
        super(FeatureExtractor, self).__init__()
        
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_p))
        
        # Second hidden layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_p))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class VariationalOrdinalGPModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)

        super(VariationalOrdinalGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel()
)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # the model returns the posterior distribution p(f*|X*,X,y)
        return latent_pred 

class DeepVariationalOrdinalGPModel(AbstractVariationalGP):
    def __init__(self, train_x, input_dim, hidden_dim, feature_dim):
        # Set up the variational distribution and strategy.
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(DeepVariationalOrdinalGPModel, self).__init__(variational_strategy)
        
        # Add a feature extractor.
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim, dropout_p=0.2, use_batchnorm=True)
        
        # Define the GP mean and covariance modules operating on the extracted features.
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        # Extract latent features using the deep network.
        features = self.feature_extractor(x)
        # Compute the mean and covariance on the latent features.
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        # Return the GP posterior as a multivariate normal.
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred