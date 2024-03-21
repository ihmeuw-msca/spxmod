import numpy as np
from regmod.prior import GaussianPrior, Prior, UniformPrior
from regmod.variable import Variable
from regmodsm.dimension import CategoricalDimension
from regmodsm.space import Space
from regmodsm._typing import DataFrame, NDArray


class VarGroup:
    """Variable group created by partitioning a variable along a dimension.

    Parameters
    ----------
    name : str
        Name of the variable column in the data.
    space : Space, optional
        Space to partition the variable on. If None, the variable is
        not partitioned.
    lam : float, optional
        Regularization parameter for the coefficients in the variable
        group. Default is 0. If the dimension is numerical, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam) is set on
        the differences between neighboring coefficients along the
        dimension. If the dimension is categorical, a Gaussian prior
        with mean 0 and standard deviation 1/sqrt(lam) is set on the
        coefficients.
    lam_mean : float, optional
        Regularization parameter for the mean of the coefficients in the
        variable group. Default is 1e-8. A Gaussian prior with mean 0
        and standard deviation 1/sqrt(lam_mean) is set on the mean of
        the coefficients.
    gprior : tuple, optional
        Gaussian prior for the variable. Default is (0, np.inf).
        Argument is overwritten with (0, 1/sqrt(lam)) if dimension is
        categorical.
    uprior : tuple, optional
        Uniform prior for the variable. Default is (-np.inf, np.inf).
    scale_by_distance : bool, optional
        Whether to scale the prior standard deviation by the distance
        between the neighboring values along the dimension. For
        numerical dimensions only. Default is False.

    TODO
    ----
    * change VarGroup to a better class name
    * change all priors to use dictionary rather than regmod class

    """

    def __init__(
        self,
        name: str,
        space: Space = Space(),
        lam: float | dict[str, float] = 0.0,
        lam_mean: float = 0.0,
        gprior: tuple[float, float] = (0.0, np.inf),
        uprior: tuple[float, float] = (-np.inf, np.inf),
        scale_by_distance: bool = False,
    ) -> None:
        self.name = name
        self.space = space
        self.lam = lam
        self.lam_mean = lam_mean
        self._gprior = gprior
        self._uprior = uprior
        self.scale_by_distance = scale_by_distance

        # transfer lam to gprior when dim is categorical
        if self.space is not None:
            if isinstance(self.lam, float):
                self.lam = {name: self.lam for name in self.space.dim_names}

            # TODO: this behavior is up-to-discussion
            lam_cat = sum(
                [
                    self.lam.get(dim.name, 0.0)
                    for dim in self.space.dims
                    if isinstance(dim, CategoricalDimension)
                ]
            )
            if lam_cat > 0:
                self._gprior = (0.0, 1.0 / np.sqrt(lam_cat))

    @property
    def gprior(self) -> GaussianPrior:
        """Gaussian prior for the variable group."""
        return GaussianPrior(mean=self._gprior[0], sd=self._gprior[1])

    @property
    def uprior(self) -> UniformPrior:
        """Uniform prior for the variable group."""
        return UniformPrior(lb=self._uprior[0], ub=self._uprior[1])

    @property
    def priors(self) -> list[Prior]:
        """List of Gaussian and Uniform priors for the variable group."""
        return [self.gprior, self.uprior]

    @property
    def size(self) -> int:
        """Number of variables in the variable group."""
        return self.space.size

    def get_variables(self) -> list[Variable]:
        """Returns the list of variables in the variable group."""
        variables = [
            Variable(name, priors=self.priors)
            for name in self.space.build_encoded_names(self.name)
        ]
        return variables

    def build_smoothing_prior(self) -> dict[str, NDArray]:
        """Returns the smoothing Gaussian prior for the variable group.

        If the dimension is numerical and lam > 0, a Gaussian prior with
        mean 0 and standard deviation 1/sqrt(lam) is set on the
        differences between neighboring coefficients along the
        dimension. For both dimension types if lam_mean > 0, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam_mean) is set
        on the mean of the coefficients.

        Returns
        -------
        dict[str, NDArray]
            Smoothing Gaussian prior matrix and vector.

        """
        return self.space.build_smoothing_prior(
            self.lam, self.lam_mean, self.scale_by_distance
        )

    def encode(self, data: DataFrame) -> DataFrame:
        """Encode variable column based on the space.

        Parameters
        ----------
        data : DataFrame
            Data containing the variable column.

        Returns
        -------
        DataFrame
            Encoded variable columns.

        """
        return self.space.encode(data, column=self.name)
