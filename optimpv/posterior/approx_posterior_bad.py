"""Module containing classes and functions for posterior analysis of parameters using the ML models from the BO optimization.
This module provides functionality to visualize the posterior distributions of parameters
using various plots, including 1D and 2D posteriors, devil's plots, and density plots."""

######### Package Imports #########################################################################
import numpy as np
import pandas as pd
from pyroapi import pyro
import seaborn as sns
import matplotlib.pyplot as plt
import itertools, torch
from scipy.special import logsumexp
from itertools import combinations
import ax
from ax import *
from ax.core.observation import ObservationFeatures
# from ax.core.base_trial import TrialStatus as T
from optimpv.general.general import inv_loss_function
from optimpv.axBOtorch.axUtils import get_df_from_ax

from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
# Create a Gaussian Process Regression model
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel, AdditiveKernel, ProductKernel,PolynomialKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize  
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.cross_validation import batch_cross_validation
from botorch.cross_validation import gen_loo_cv_folds
from sklearn.metrics import mean_squared_error, r2_score
from itertools import chain
import time
import botorch
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import arviz as az
######### Function Definitions ####################################################################

class ApproximatePosterior:



    def __init__(self, params, df, sigma, outcome_name, loss, n, **kwargs):
        """_summary_

        Parameters
        ----------
        params : list of Fitparam() objects, optional
            List of Fitparam() objects, by default None
        df : pd.DataFrame
            DataFrame containing the optimization results.
        outcome_name : str
            Name of the outcome variable to analyze.
        loss : str
            Loss function used in the optimization.
        """        
        
        self.params = params
        self.df = df
        self.sigma = sigma
        self.outcome_name = outcome_name
        self.loss = loss
        self.n = n
        self.kwargs = kwargs
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.num_free_params = sum(1 for p in self.params if not p.type == 'fixed')
        self.bounds_tensor = self.get_bounds_tensor()
        max_size_cv = kwargs.get('max_size_cv', 700)
        if len(self.df) > max_size_cv:
            self.do_batch_cv = False
        else:
            self.do_batch_cv = True
        
        # convert sigma to tensor
        if not isinstance(sigma, torch.Tensor) and sigma.device != self.device:
            self.sigma = torch.tensor(sigma).to(self.device)
        

    def get_bounds_tensor(self):
        """Get the bounds of the parameters.

        Returns
        -------
        bounds : torch.Tensor
            Tensor containing the bounds of the parameters.
        """
        
        # ignore the fixed parameters and take log10 of the bounds if force_log is True
        bounds = []
        for p in self.params:
            if not p.type == 'fixed':
                if p.force_log:
                    bounds.append([np.log10(p.bounds[0]), np.log10(p.bounds[1])])
                else:
                    bounds.append([p.bounds[0], p.bounds[1]])

        return torch.tensor(bounds).reshape(2, self.num_free_params).to(self.device)
    

    def get_Xy_train_tensor(self):
        """Get the training data as tensors.

        Returns
        -------
        X_train : torch.Tensor
            Tensor containing the training input data.
        y_train : torch.Tensor
            Tensor containing the training output data.
        """
        
        X_train = []
        y_train = []
        for index, row in self.df.iterrows():
            x_row = []
            for p in self.params:
                if not p.type == 'fixed':
                    x_row.append(row[p.name])
            X_train.append(x_row)
            y_train.append(row[self.outcome_name])

        X_train = torch.tensor(X_train).to(self.device).reshape(-1, self.num_free_params)
        y_train = torch.tensor(y_train).to(self.device).unsqueeze(-1)
        
        return X_train, y_train

    def train_surrogate_model(self):
        """Train a surrogate model on the optimization data.

        Parameters
        ----------
        model : ax.models.torch.botorch.BotorchModel
            The surrogate model to train.
        training_iterations : int, optional
            Number of training iterations, by default 1000
        """

        # get the data from the dataframe
        df = self.df
        outcome_name = self.outcome_name

        # get the training data
        X_train, y_train = self.get_Xy_train_tensor()

        input_transform = Normalize(d=X_train.shape[-1], bounds=self.bounds_tensor)

        if self.do_batch_cv:
            cv_folds = gen_loo_cv_folds(X_train, y_train)

            outcome_transform = Standardize(m=1)
            outcome_transform._batch_shape = cv_folds.train_Y.shape[:-2]

            # composite_kernel = ProductKernel(
            #     MaternKernel(nu=2.5, ard_num_dims=self.num_free_params),
            #      LinearKernel(ard_num_dims=self.num_free_params)
            # )
            # composite_kernel = ProductKernel(
            #     MaternKernel(nu=2.5, ard_num_dims=self.num_free_params),
            #      PolynomialKernel(power=2, ard_num_dims=self.num_free_params)
            # )
            # Perform batch cross-validation
            cv_results = batch_cross_validation(
                model_cls= SingleTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                model_init_kwargs={
                    "input_transform": input_transform,
                    "outcome_transform": outcome_transform,
                    "covar_module": ScaleKernel(base_kernel=MaternKernel(nu=2.5, ard_num_dims=self.num_free_params)),
                    # "covar_module": ScaleKernel(composite_kernel),
                    "likelihood": GaussianLikelihood(),
                }
            )
            self.cv_results = cv_results
            self.cv_folds = cv_folds
            self.model = cv_results.model
        else:
            # Fit the model on the entire dataset
            gp_model = SingleTaskGP(
                train_X=X_train,
                train_Y=y_train,
                input_transform=input_transform,
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(base_kernel=MaternKernel(nu=2.5, ard_num_dims=self.num_free_params)),
                # covar_module=ScaleKernel(composite_kernel),
                likelihood=GaussianLikelihood(),
            ).to(self.device)

            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(self.device)
            fit_gpytorch_mll(mll)


            self.model = gp_model

    def plot_cv_results(self):
        """Plot the cross-validation results."""

        posterior = self.cv_results.posterior
        mean = posterior.mean
        cv_error = ((self.cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()

        print(f"Cross-validation error: {cv_error : 4.2}")

        # Get confidence intervals
        lower, upper = posterior.mvn.confidence_region()

        # Create parity plot
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.plot([self.cv_folds.test_Y.cpu().min(), self.cv_folds.test_Y.cpu().max()], 
                [self.cv_folds.test_Y.cpu().min(), self.cv_folds.test_Y.cpu().max()], 
                "k", label="true objective", linewidth=2)

        axes.set_xlabel("Actual")
        axes.set_ylabel("Predicted")

        # Plot predictions with error bars
        axes.errorbar(
            x=self.cv_folds.test_Y.cpu().numpy().flatten(),
            y=mean.cpu().numpy().flatten(),
            yerr=((upper - lower) / 2).cpu().numpy().flatten(),
            fmt="*",
        )

        # Calculate and display metrics
        r2 = r2_score(self.cv_folds.test_Y.cpu().numpy().reshape(-1), mean.cpu().numpy().flatten())
        axes.set_title(f"R2: {r2:.2f}, CV Error: {cv_error:.3f}")

    def pyro_model(self, gpr_model, n, loss, sigma, device):

        params_tensor = []
        for idx, p in enumerate(self.params):
            if p.type != 'fixed':
                param_name = p.name
                bound = p.bounds
                if p.force_log:
                    bound = [np.log10(bound[0]), np.log10(bound[1])]
                params_tensor.append(pyro.sample(param_name, dist.Uniform(bound[0], bound[1])))
        params_tensor = torch.stack(params_tensor).to(device)

        # MSE = gpr_model.posterior(params_tensor.unsqueeze(0)).mean
        log_likelihood = gpr_model.posterior(params_tensor.unsqueeze(0)).mean
        # beta = 1/(sigma.mean())**2
        # beta = sigma**2
        # # sum
        # beta = beta.sum()
        # log_likelihood = - 1/2 * n * (inv_loss_function(MSE, loss)) - 1/2 * torch.log(beta) - 1/2 * torch.log(torch.tensor(2 * np.pi))
        
        # beta = 1/np.mean(sigma.cpu().numpy()**2)
        # n = 111
        # print(params_tensor,   inv_loss_function(y_pred, loss))
        # log_likelihood = -beta/2 * inv_loss_function(MSE, loss) - n/2*np.log(beta) - n/2 * np.log(2 * np.pi)
        # log_likelihood = -beta/2 * inv_loss_function(MSE, loss) - n/2*np.log(beta) - 1/2 * np.log(2 * np.pi)

        # clean
        # beta2 = sigma**2
        # beta = 1/(beta2)
        # beta = beta.sum()
        # # log_likelihood = -beta/2 * inv_loss_function(MSE, loss) #- n/2 * torch.log(torch.tensor(2 * np.pi)) + torch.log(beta)/2
        # beta2 = torch.log(beta2)
        # beta2 = beta2.sum()
        # log_likelihood = -beta/2 * inv_loss_function(MSE, loss) - n/2 * torch.log(torch.tensor(2 * np.pi)) - beta2/2

        # works 
        # beta2 = sigma**2
        # beta2 = beta2.mean()
        # beta = 1/(beta2)
        # log_likelihood = -beta/2 * inv_loss_function(MSE, loss) - n/2*torch.log(beta) - n/2 * torch.log(torch.tensor(2 * np.pi))

        pyro.factor("log_likelihood", log_likelihood)

    def run_mcmc(self, num_samples=1000, warmup_steps=200, num_chains=1):
        """Run MCMC to sample from the posterior distribution of the parameters.

        Parameters
        ----------
        num_samples : int, optional
            Number of MCMC samples to draw, by default 1000
        warmup_steps : int, optional
            Number of warmup steps, by default 200
        num_chains : int, optional
            Number of MCMC chains, by default 1
        """

        device = self.device
        # X_train, y_train = self.get_Xy_train_tensor()
        n = self.n
        # free cuda memory
        # torch.cuda.empty_cache()
        # train the surrogate model
        self.train_surrogate_model()
        gpr_model = self.model

        # define the pyro model
        samples = None
        for i in range(num_chains):
            rng_seed = np.random.randint(1,1e6)
            pyro.set_rng_seed(rng_seed)
            nuts_kernel = NUTS(self.pyro_model)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
            mcmc.run(gpr_model, n, self.loss, self.sigma, device)
            if samples is None:
                samples = mcmc.get_samples()
            else:
                for key in samples.keys():
                    samples[key] = torch.cat([samples[key], mcmc.get_samples()[key]], dim=0)

        self.mcmc_samples = samples

    def arviz_pairplot(self):
        """Plot the pairplot of the MCMC samples using ArviZ."""

        

        data = az.from_pyro(mcmc_samples=self.mcmc_samples)
        az.plot_pair(data, kind='kde', marginals=True)
        plt.show()

if __name__ == "__main__":
    pass