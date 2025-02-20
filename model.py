import torch

import gpytorch
import botorch


class ExactProbitGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        likelihood = likelihood.to(train_targets)

        super().__init__(train_inputs, train_targets, likelihood)

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        if self.training:
            res = super.__call__(*inputs, **kwargs)

        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            full_inputs = [torch.cat([train_input, ii], dim=-2) for train_input, ii in zip(train_inputs, inputs)]
            full_output = super().__call__(*full_inputs, **kwargs)

            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            return gpytorch.distributions.MultivariateNormal(predictive_mean, predictive_covar)
