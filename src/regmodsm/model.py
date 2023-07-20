from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel
from regmod.models import Model as RegmodModel
from regmod.models import PoissonModel
from regmod.prior import LinearGaussianPrior, UniformPrior
from regmod.variable import Variable
from scipy.linalg import block_diag
from scipy.stats import norm

_model_dict = {
    "binomial": BinomialModel,
    "poisson": PoissonModel,
    "gaussian": GaussianModel,
}


class Model:
    def __init__(
        self,
        model_type: str,
        obs: str,
        covs: list[str],
        dims: Optional[dict[str, str]] = None,
        lams: Optional[dict[str, float]] = None,
        coef_bounds: Optional[dict[str, tuple[float, float]]] = None,
        weights: str = "weight",
        default_dim: Optional[str] = None,
        default_lam: float = 1.0,
        param_specs: Optional[dict] = None,
    ) -> None:
        dims = dims or {}
        lams = lams or {}
        coef_bounds = coef_bounds or {}
        param_specs = param_specs or {}

        for cov in covs:
            if cov not in dims:
                dims[cov] = default_dim
            if cov not in lams:
                lams[cov] = default_lam

        self.model_type = model_type
        self.obs = obs
        self.covs = covs
        self.dims = dims
        self.lams = lams
        self.coef_bounds = coef_bounds
        self.weights = weights
        self.param_specs = param_specs

        self._dim_vals: dict[str, NDArray] = {}
        self._model: Optional[RegmodModel] = None

    def _get_dim_vals(self, data: DataFrame) -> dict[str, NDArray]:
        dim_vals = {}
        for cov in self.covs:
            dim = self.dims[cov]
            if (dim is not None) and (dim not in dim_vals):
                dim_vals[dim] = np.unique(data[dim])
        return dim_vals

    def _get_smoothing_prior(self) -> tuple[NDArray, NDArray]:
        prior_mats = []
        prior_vecs = []
        for cov in self.covs:
            dim = self.dims[cov]
            lam = self.lams[cov]
            if dim is not None:
                n = len(self._dim_vals[dim])
                delta = np.diff(self._dim_vals[dim])
                delta = delta / delta.min()
                mat = np.zeros(shape=(n - 1, n))
                id0 = np.diag_indices(n - 1)
                id1 = (id0[0], id0[1] + 1)
                mat[id0], mat[id1] = -1.0, 1.0
                vec = np.zeros(shape=(2, n - 1))
                vec[1] = 1 / np.sqrt(lam) if lam > 0 else np.inf
            else:
                mat = np.zeros(shape=(0, 1))
                vec = np.zeros(shape=(2, 0))
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
        for cov in self.covs:
            dim = self.dims[cov]
            coef_bounds = self.coef_bounds.get(cov, (-np.inf, np.inf))
            if dim is not None:
                for i in range(len(self._dim_vals[dim])):
                    uprior = UniformPrior(lb=coef_bounds[0], ub=coef_bounds[1])
                    variables.append(Variable(f"{cov}_{i}", priors=[uprior]))
            else:
                uprior = UniformPrior(lb=coef_bounds[0], ub=coef_bounds[1])
                variables.append(Variable(cov, priors=[uprior]))

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

        for cov in self.covs:
            dim = self.dims[cov]
            if dim is not None:
                cov_cols = (
                    data[[dim, cov]]
                    .pivot(columns=dim, values=cov)
                    .fillna(0.0)
                    .rename(
                        columns={
                            val: f"{cov}_{i}"
                            for i, val in enumerate(self._dim_vals[dim])
                        }
                    )
                )
                data.drop(columns=cov, inplace=True)
                data = pd.concat([data, cov_cols], axis=1)

        return data

    def fit(self, data: DataFrame, **optimizer_options) -> None:
        self._dim_vals = self._get_dim_vals(data)
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
        data = self._expand_data(data)
        self._model.data.attach_df(data)
        param = self._model.params[0]
        coef = self._model.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = param.get_mat(self._model.data)
        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = self._model.opt_vcov
            lin_param_sd = np.sqrt((mat.dot(vcov) * mat).sum(axis=1))
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
    """Detach data and all the arrays from the regmod model."""
    model.data.detach_df()
    del model.mat
    del model.uvec
    del model.gvec
    del model.linear_uvec
    del model.linear_gvec
    del model.linear_umat
    del model.linear_gmat

    return model
