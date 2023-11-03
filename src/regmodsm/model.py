from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel
from regmod.models import Model as RegmodModel
from regmod.models import PoissonModel
from regmod.prior import LinearGaussianPrior, UniformPrior, GaussianPrior, Prior
from regmod.variable import Variable
from scipy.linalg import block_diag
from scipy.stats import norm
from regmodsm.linalg import get_pred_var

_model_dict = {
    "binomial": BinomialModel,
    "poisson": PoissonModel,
    "gaussian": GaussianModel,
}


class Dimension:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.vals = None

    def set_vals(self, data: DataFrame) -> None:
        self.vals = list(np.unique(data[self.name]))


class VarGroup:
    def __init__(
        self,
        col: str,
        dim: Optional[Dimension] = None,
        lam: float = 0.0,
        gprior: tuple[float, float] = (0.0, np.inf),
        uprior: tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        self.col = col
        self.dim = dim
        self.lam = lam
        self._gprior = gprior
        self._uprior = uprior

    @property
    def gprior(self) -> GaussianPrior:
        return GaussianPrior(mean=self._gprior[0], sd=self._gprior[1])

    @property
    def uprior(self) -> UniformPrior:
        return UniformPrior(lb=self._uprior[0], ub=self._uprior[1])

    @property
    def priors(self) -> list[Prior]:
        return [self.gprior, self.uprior]

    @property
    def size(self) -> int:
        if self.dim is None:
            return 1
        if self.dim.vals is None:
            raise ValueError(f"Please set values in dim={self.dim.name} first")
        return len(self.dim.vals)

    def get_variables(self) -> list[Variable]:
        if self.dim is None:
            return [Variable(self.col, priors=self.priors)]
        variables = [
            Variable(f"{self.col}_{self.dim.name}_{i}", priors=self.priors)
            for i in range(len(self.dim.vals))
        ]
        return variables

    def get_smoothing_gprior(self,mean_lam = 1e-3) -> tuple[NDArray, NDArray]:
        """
        mean_lam regularizes the mean of all the coefficients
        """
        if self.dim is None or self.lam == 0.0:
            return np.empty(shape=(0, self.size)), np.empty(shape=(2, 0))
        
        if self.dim.type == "categorical":
            n = len(self.dim.vals)
            vec = np.zeros(shape=(2, n))
            vec[1] = 1 / np.sqrt(self.lam)
            return np.identity(self.size),vec
        
        n = len(self.dim.vals)
        delta = np.diff(self.dim.vals) #Delta is unused here, I'll not touch it for now.
        delta = delta / delta.min()
        mat = np.zeros(shape=(n, n)) #Would be (n-1)x(n), but we have an extra row at bottom for ones
        id0 = np.diag_indices(n - 1)
        id1 = (id0[0], id0[1] + 1)
        mat[id0], mat[id1] = -1.0, 1.0
        mat[-1] = 1/n
        vec = np.zeros(shape=(2, n))
        vec[1,:-1] = 1 / np.sqrt(self.lam)
        vec[1,-1]= 1 / np.sqrt(mean_lam)
        
        return mat, vec

    def expand_data(self, data: DataFrame) -> DataFrame:
        if self.dim is None:
            return DataFrame(index=data.index)
        df_vars = pd.get_dummies(data[self.dim.name], sparse=True).mul(
            data[self.col], axis=0
        )
        df_vars.rename(
            columns={
                val: f"{self.col}_{self.dim.name}_{i}"
                for i, val in enumerate(self.dim.vals)
            },
            inplace=True,
        )
        df_vars.drop(
            columns=[
                col
                for col in df_vars.columns
                if (not isinstance(col, str)) or (not col.startswith(self.col))
            ],
            inplace=True,
        )
        return df_vars


class Model:
    def __init__(
        self,
        model_type: str,
        obs: str,
        dims: list[dict],
        var_groups: list[dict],
        weights: str = "weight",
        param_specs: Optional[dict] = None,
    ) -> None:
        self.model_type = model_type
        self.obs = obs

        self.dims = tuple(map(lambda kwargs: Dimension(**kwargs), dims))
        self._dim_dict = {dim.name: dim for dim in self.dims}

        for var_group in var_groups:
            if ("dim" in var_group) and (var_group["dim"] is not None):
                var_group["dim"] = self._dim_dict[var_group["dim"]]
        self.var_groups = tuple(map(lambda kwargs: VarGroup(**kwargs), var_groups))

        self.weights = weights
        self.param_specs = param_specs or {}

        self._model: Optional[RegmodModel] = None

    def _set_dim_vals(self, data: DataFrame) -> None:
        for dim in self.dims:
            dim.set_vals(data)

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

    def fit(self, data: DataFrame, **optimizer_options) -> None:
        self._set_dim_vals(data)
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
