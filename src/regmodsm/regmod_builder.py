from regmod.data import Data
from regmod.variable import Variable
from regmod.prior import LinearGaussianPrior, GaussianPrior, UniformPrior
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmodsm._typing import RegmodModel


_model_dict = {
    "binomial": BinomialModel,
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
