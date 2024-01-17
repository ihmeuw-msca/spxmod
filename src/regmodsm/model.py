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
from sklearn.preprocessing import OneHotEncoder

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
        self.encoder = OneHotEncoder()
        self.encoder.fit(data[[self.name]])



class VarGroup:
    def __init__(
        self,
        col: str,
        dim: Optional[Dimension] = None,
        lam: float = 0.0,
        lam_mean: float = 1e-8,
        gprior: tuple[float, float] = (0.0, np.inf),
        uprior: tuple[float, float] = (-np.inf, np.inf),
        scale_by_distance: bool = False,
    ) -> None:
        self.col = col
        self.dim = dim
        self.lam = lam
        self.lam_mean = lam_mean
        self._gprior = gprior
        self._uprior = uprior
        self.scale_by_distance = scale_by_distance

        # transfer lam to gprior when dim is categorical
        if self.dim.type == "categorical" and self.lam > 0.0:
            self._gprior = (0.0, 1.0 / np.sqrt(self.lam))

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

    def get_smoothing_gprior(self) -> tuple[NDArray, NDArray]:
        """
        mean_lam regularizes the mean of all the coefficients
        """
        n = self.size
        mat = np.empty(shape=(0, n))
        vec = np.empty(shape=(2, 0))

        if self.dim.type == "continuous" and self.lam > 0.0:
            mat = np.zeros(shape=(n - 1, n))
            id0 = np.diag_indices(n - 1)
            id1 = (id0[0], id0[1] + 1)
            mat[id0], mat[id1] = -1.0, 1.0
            vec = np.zeros(shape=(2, n - 1))
            vec[1] = 1 / np.sqrt(self.lam)
            if self.scale_by_distance:
                delta = np.diff(self.dim.vals)
                delta /= delta.min()
                vec[1] *= delta

        if self.lam_mean > 0.0:
            mat = np.vstack([mat, np.repeat(1.0 / n, n)])
            vec = np.hstack([vec, np.array([[0.0], [1.0 / np.sqrt(self.lam_mean)]])])

        return mat, vec

    def expand_data(self, data: DataFrame) -> DataFrame:
        if self.dim is None:
            return DataFrame(index=data.index)
        
        dummies = pd.DataFrame.sparse.from_spmatrix(
            self.dim.encoder.transform(data[[self.dim.name]]),
            columns = self.dim.encoder.categories_[0]
            )
                
        df_vars = dummies.mul(
            data.reset_index()[self.col], axis=0
        ).set_index(data.index)

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

    def fit(self, data: DataFrame, dim_val_data=None, **optimizer_options) -> None:
        if dim_val_data is None:
            self._set_dim_vals(data)
        else:
            self._set_dim_vals(dim_val_data)
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
