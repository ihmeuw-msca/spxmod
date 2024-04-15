import numpy as np
import scipy.sparse as sp
from msca.linalg.matrix import Matrix, asmatrix
from msca.optim.solver import IPSolver, NTSolver
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmod.prior import GaussianPrior, LinearGaussianPrior, UniformPrior
from regmod.variable import Variable

from spxmod.typing import Callable, NDArray, RegmodModel


def msca_optimize(
    model: RegmodModel, x0: NDArray | None = None, options: dict | None = None
) -> NDArray:
    x0 = np.zeros(model.size) if x0 is None else x0
    options = options or {}

    if model.cmat.size == 0:
        solver = NTSolver(model.objective, model.gradient, model.hessian)
    else:
        solver = IPSolver(
            model.objective, model.gradient, model.hessian, model.cmat, model.cvec
        )
    result = solver.minimize(x0=x0, **options)
    model.opt_result = result
    model.opt_coefs = result.x.copy()
    model.opt_hessian = model.hessian(model.opt_coefs)
    model.opt_jacobian2 = model.jacobian2(model.opt_coefs)
    return result.x


class SparseBinomialModel(BinomialModel):
    def hessian_from_gprior(self) -> Matrix:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        Matrix
            Hessian matrix.
        """
        hess = sp.diags(1.0 / self.gvec[1] ** 2)
        if self.linear_gvec.size > 0:
            hess += (self.linear_gmat.T.scale_cols(1.0 / self.linear_gvec[1] ** 2)).dot(
                self.linear_gmat
            )
        return asmatrix(hess)

    def fit(self, optimizer: Callable = msca_optimize, **optimizer_options) -> None:
        super().fit(optimizer=optimizer, **optimizer_options)


_model_dict = {
    "binomial": SparseBinomialModel,
    "gaussian": GaussianModel,
    "poisson": PoissonModel,
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
