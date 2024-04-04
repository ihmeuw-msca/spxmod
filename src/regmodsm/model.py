"""Fit a RegMod model with variables regularized in dimension groupings.

Examples
--------
Fit a RegMod model with covariate values smoothed by age. In this model,
a different coefficient for the covariate `mean_BMI` is fit for each
unique value of `age_mid`. Because the age dimension is numerical, a
Gaussian prior with mean 0 and standard deviation 1/sqrt(lam) is set on
the differences between neighboring coefficients. This ensures that
coefficients vary smoothly by age. The uniform prior forces the
coefficients to have positive values.

>>> import numpy as np
>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": "age_mid", "type": "numerical"}],
        var_groups=[{"col": "mean_BMI", "dim": "age_mid", "lam": 1.0, "uprior": (0.0, np.inf)}],
        weights="sample_size"
    )

Fit a RegMod model with intercept values smoothed by region. In this
model, a different intercept is fit for each unique value of
`region_id`. A Gaussian prior with mean 0 and standard deviation
1/sqrt(lam) is set on the mean of the intercept values.

>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": "region_id", "type": "categorical"}],
        var_groups=[{"col": "intercept", "dim": "region_id", "lam_mean": 1.0}],
        weights="sample_size"
    )

Fit a RegMod model with intercept values smoothed by age-year. In this
model, a different intercept is fit for each unique age_group_id-year_id
pair.

>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": ["age_group_id", "year_id"], "type": 2*["categorical"]}],
        var_groups=[{"col": "intercept", "dim": "age_group_id*year_id"}],
        "weights"="sample_size"
    )

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmod.models import Model as RegmodModel
from regmod.prior import LinearGaussianPrior
from scipy.linalg import block_diag
from scipy.stats import norm

from regmodsm.linalg import get_pred_var
from regmodsm.dimension import Dimension
from regmodsm.vargroup import VarGroup

_model_dict = {
    "binomial": BinomialModel,
    "gaussian": GaussianModel,
    "poisson": PoissonModel,
}

class Model:
    """RegMod Smooth model.

    Parameters
    ----------
    model_type : {"binomial", "poisson", "gaussian"}
        RegMod model type.
    obs : str
        Name of the observation column in the data.
    dims : list[dict]
        List of dictionaries containing dimension names and arguments.
    var_groups : list[dict]
        List of dictionaries containing variable group names and arguments.
    weights : str, optional
        Name of the weight column in the data. Default is "weight".
    param_specs : dict, optional
        Additional parameter specifications for the model.

    """

    def __init__(
        self,
        model_type: str,
        obs: str,
        dims: list[dict],
        var_groups: list[dict],
        weights: str = "weight",
        param_specs: dict | None = None,
    ) -> None:
        self.model_type = model_type
        self.obs = obs

        self.dims = tuple(map(lambda kwargs: Dimension(**kwargs), dims))
        self._dim_dict = {dim.label: dim for dim in self.dims}

        for var_group in var_groups:
            if ("dim" in var_group) and (var_group["dim"] is not None):
                var_group["dim"] = self._dim_dict[var_group["dim"]]
        self.var_groups = tuple(map(lambda kwargs: VarGroup(**kwargs), var_groups))

        self.weights = weights
        self.param_specs = param_specs or {}

        self._model: RegmodModel | None = None

    def _set_dim_span(self, data: DataFrame) -> None:
        for dim in self.dims:
            dim.set_span(data)

    def _get_smoothing_prior(self) -> tuple[NDArray, NDArray]:
        prior_mats = []
        prior_vecs = []
        for var_group in self.var_groups:
            mat, vec = var_group.get_smoothing_gprior()
            prior_mats.append(mat)
            prior_vecs.append(vec)
        prior_mat = block_diag(*prior_mats)
        prior_vec = np.hstack(prior_vecs)
        return prior_mat, prior_vec

    def _get_model(self) -> RegmodModel:
        data = Data(
            col_obs=self.obs,
            col_weights=self.weights,
        )

        variables = []
        for var_group in self.var_groups:
            variables.extend(var_group.get_variables())

        linear_gpriors = []
        prior_mat, prior_vec = self._get_smoothing_prior()
        if len(prior_mat) > 0:
            linear_gpriors = [
                LinearGaussianPrior(mat=prior_mat, mean=prior_vec[0], sd=prior_vec[1])
            ]

        model_class = _model_dict[self.model_type]
        model_param = model_class.param_names[0]

        model = model_class(
            data,
            param_specs={
                model_param: {
                    "variables": variables,
                    "linear_gpriors": linear_gpriors,
                    **self.param_specs,
                }
            },
        )
        return model

    def _expand_data(self, data: DataFrame) -> DataFrame:
        data = data.copy()
        data["intercept"] = 1.0

        for var_group in self.var_groups:
            df_covs = var_group.expand_data(data)
            data = pd.concat([data, df_covs], axis=1)

        return data

    def fit(
        self,
        data: DataFrame,
        data_dim_vals: DataFrame | None = None,
        **optimizer_options,
    ) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        data : DataFrame
            Data to fit the model to.
        data_dim_vals : DataFrame, optional
            Data containing the unique dimension values. If None, values
            are extracted from the data. Default is None.
        optimizer_options
            Additional options for the optimizer.

        """
        if data_dim_vals is None:
            self._set_dim_span(data)
        else:
            self._set_dim_span(data_dim_vals)
        self._model = self._get_model()
        data = self._expand_data(data)
        self._model.attach_df(data)
        self._model.fit(**optimizer_options)
        _detach_df(self._model)

    def predict(
        self,
        data: DataFrame,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        """Predict the response variable.

        Parameters
        ----------
        data : DataFrame
            Data to predict the response variable.
        return_ui : bool, optional
            Whether to return the prediction interval. Default is False.
        alpha : float, optional
            Significance level for the prediction interval. Default is 0.05.

        Returns
        -------
        NDArray
            Predicted response variable. If `return_ui` is True, the prediction interval
            is also returned.

        """
        data = self._expand_data(data)
        self._model.data.attach_df(data)
        param = self._model.params[0]
        coef = self._model.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = np.ascontiguousarray(param.get_mat(self._model.data))

        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = self._model.opt_vcov
            lin_param_sd = np.sqrt(get_pred_var(mat, vcov))
            lin_param_lower = norm.ppf(0.5 * alpha, loc=lin_param, scale=lin_param_sd)
            lin_param_upper = norm.ppf(
                1 - 0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            pred = np.vstack(
                [
                    pred,
                    param.inv_link.fun(lin_param_lower),
                    param.inv_link.fun(lin_param_upper),
                ]
            )
        self._model.data.detach_df()
        return pred


def _detach_df(model: RegmodModel) -> RegmodModel:
    """Detach input data and model arrays from the RegMod model."""
    model.data.detach_df()
    del model.mat
    del model.uvec
    del model.gvec
    del model.linear_uvec
    del model.linear_gvec
    del model.linear_umat
    del model.linear_gmat

    return model
