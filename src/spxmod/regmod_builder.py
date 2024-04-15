import numpy as np
import scipy.sparse as sp
from msca.linalg.matrix import Matrix, asmatrix
from msca.optim.solver import IPSolver, NTSolver
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmod.prior import GaussianPrior, LinearGaussianPrior, UniformPrior
from regmod.variable import Variable
from scipy.stats import norm

from spxmod.linalg import get_pred_var
from spxmod.typing import Callable, DataFrame, NDArray, RegmodModel


def msca_optimize(
    model: RegmodModel, x0: NDArray | None = None, options: dict | None = None
) -> NDArray:
    x0 = np.zeros(model.size) if x0 is None else x0
    options = options or {}

    if model.cmat.size == 0:
        solver = NTSolver(model.objective, model.gradient, model.hessian)
    else:
        solver = IPSolver(
            model.objective,
            model.gradient,
            model.hessian,
            model.cmat,
            model.cvec,
        )
    result = solver.minimize(x0=x0, **options)
    model.opt_result = result
    model.opt_coefs = result.x.copy()
    model.opt_hessian = model.hessian(model.opt_coefs)
    model.opt_jacobian2 = model.jacobian2(model.opt_coefs)
    return result.x


class SparseRegmodModel(RegmodModel):
    def attach_df(self, df: DataFrame, encode: Callable) -> None:
        self.data.attach_df(df)
        self.mat = [asmatrix(sp.csc_matrix(encode(df)))]
        self.uvec = self.get_uvec()
        self.gvec = self.get_gvec()
        self.linear_uvec = self.get_linear_uvec()
        self.linear_gvec = self.get_linear_gvec()
        self.linear_umat = asmatrix(sp.csc_matrix((0, self.size)))
        self.linear_gmat = asmatrix(sp.csc_matrix((0, self.size)))
        param = self.params[0]
        if self.params[0].linear_gpriors:
            self.linear_gmat = asmatrix(
                sp.csc_matrix(param.linear_gpriors[0].mat)
            )
        if self.params[0].linear_upriors:
            self.linear_umat = asmatrix(
                sp.csc_matrix(param.linear_upriors[0].mat)
            )

        # parse constraints
        cmat = asmatrix(
            sp.vstack(
                [sp.identity(self.mat[0].shape[1]), self.linear_umat],
                format="csr",
            )
        )
        cvec = np.hstack([self.uvec, self.linear_uvec])

        if cmat.size > 0:
            scale = abs(cmat).max(axis=1).toarray().ravel()
            valid = ~np.isclose(scale, 0.0)
            cmat, cvec, scale = cmat[valid], cvec[:, valid], scale[valid]
            if scale.size > 0:
                cmat = cmat.scale_rows(1.0 / scale)
                cvec = cvec / scale

                neg_valid = ~np.isneginf(cvec[0])
                pos_valid = ~np.isposinf(cvec[1])
                cmat = sp.vstack(
                    [-cmat[neg_valid], cmat[pos_valid]], format="csr"
                )
                cvec = np.hstack([-cvec[0][neg_valid], cvec[1][pos_valid]])

        self.cmat = asmatrix(sp.csc_matrix(cmat))
        self.cvec = cvec

    def detach_df(self) -> None:
        self.data.detach_df()
        del self.mat
        del self.uvec
        del self.gvec
        del self.linear_uvec
        del self.linear_gvec
        del self.linear_umat
        del self.linear_gmat
        del self.cmat
        del self.cvec

    def hessian_from_gprior(self) -> Matrix:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        Matrix
            Hessian matrix.

        """
        hess = sp.diags(1.0 / self.gvec[1] ** 2, format="csr")
        if self.linear_gvec.size > 0:
            hess += (
                self.linear_gmat.T.scale_cols(1.0 / self.linear_gvec[1] ** 2)
            ).dot(self.linear_gmat)
        return asmatrix(hess)

    def fit(
        self,
        data: DataFrame,
        encode: Callable,
        optimizer: Callable = msca_optimize,
        **optimizer_options,
    ) -> None:
        self.attach_df(data, encode)
        super().fit(optimizer=optimizer, **optimizer_options)
        self.detach_df()

    def predict(
        self,
        data: DataFrame,
        encode: Callable,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        mat = sp.csc_matrix(encode(data))
        param, coef = self.params[0], self.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            # TODO: explore the sparsity of the variance-covariance matrix
            vcov = get_vcov(self.opt_hessian, self.opt_jacobian2)
            lin_param_sd = np.sqrt(get_pred_var(mat, vcov))
            lin_param_lower = norm.ppf(
                0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
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
        return pred


class SparseBinomialModel(SparseRegmodModel, BinomialModel):
    """Sparse Binomial model."""


class SparseGaussianModel(SparseRegmodModel, GaussianModel):
    """Sparse Gaussian model."""


class SparsePoissonModel(SparseRegmodModel, PoissonModel):
    """Sparse Poisson model."""


_model_dict = {
    "binomial": SparseBinomialModel,
    "gaussian": SparseGaussianModel,
    "poisson": SparsePoissonModel,
}


def build_regmod_model(
    model_type: str,
    data: dict,
    variables: list[dict],
    linear_gpriors: list[dict],
    param_specs: dict,
) -> RegmodModel:
    # build data
    data = Data(**data)

    # build variables
    variables = [_build_regmod_variable(**kwargs) for kwargs in variables]

    # build smoothing prior
    for i, prior in enumerate(linear_gpriors):
        if prior["mat"].size > 0:
            linear_gpriors[i] = LinearGaussianPrior(**prior)

    # buid regmod model
    model_class = _model_dict[model_type]
    model_param = model_class.param_names[0]

    return model_class(
        data,
        param_specs={
            model_param: {
                "variables": variables,
                "linear_gpriors": linear_gpriors,
                **param_specs,
            }
        },
    )


def _build_regmod_variable(name: str, gprior: dict, uprior: dict) -> Variable:
    gprior = GaussianPrior(**gprior)
    uprior = UniformPrior(**uprior)
    return Variable(name=name, priors=[gprior, uprior])


def get_vcov(hessian: Matrix, jacobian2: Matrix) -> NDArray:
    hessian, jacobian2 = hessian.to_numpy(), jacobian2.to_numpy()

    # inverse hessian
    eig_vals, eig_vecs = np.linalg.eigh(hessian)
    if np.isclose(eig_vals, 0.0).any():
        raise ValueError(
            "singular Hessian matrix, please add priors or "
            "reduce number of variables"
        )
    inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

    # inspect jacobian2
    eig_vals = np.linalg.eigvalsh(jacobian2)
    if np.isclose(eig_vals, 0.0).any():
        raise ValueError(
            "singular Jacobian matrix, please add priors or "
            "reduce number of variables"
        )

    vcov = inv_hessian.dot(jacobian2)
    vcov = inv_hessian.dot(vcov.T)

    return vcov
