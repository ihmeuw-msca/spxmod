from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.sparse import coo_matrix, identity, kron, vstack

from spxmod.dimension import Dimension, NumericalDimension, build_dimension
from spxmod.typing import DataFrame, NDArray


class Space:
    """Space class host dimension information. Variable will be defined in the
    space.

    Parameters
    ----------
    name : str, optional
        Name of the space. Default is None. If None, the name is set to the
        product of the dimension names.
    dims : list of dict, optional
        List of dimension information. Default is None. If None, the dimension
        list will be empty.

    """

    def __init__(
        self,
        name: str | None = None,
        dims: list[Dimension] | None = None,
    ) -> None:
        self.dims = dims or []
        self.name = "*".join(self.dim_names) if name is None else name
        self._span: DataFrame | None = None

        if not self.dims:
            self.set_span(DataFrame())

    @classmethod
    def from_config(cls, config: dict) -> Space:
        if "dims" in config:
            config["dims"] = [build_dimension(**dim) for dim in config["dims"]]
        return cls(**config)

    @property
    def span(self) -> DataFrame:
        """The grid of the space."""
        if self._span is None:
            raise ValueError("Space span are not set.")
        return self._span

    @property
    def size(self) -> int:
        """Size of the space grid."""
        return len(self.span)

    @property
    def dim_names(self) -> list[str]:
        """Names of the dimensions."""
        return [dim.name for dim in self.dims]

    @property
    def dim_sizes(self) -> list[int]:
        """Sizes of the dimensions."""
        return [dim.size for dim in self.dims]

    def set_span(self, data: DataFrame) -> None:
        """Set the unique dimension values and the grid of space.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        for dim in self.dims:
            dim.set_span(data)
        grid = itertools.product(*[dim.span for dim in self.dims])
        self._span = DataFrame(data=grid, columns=self.dim_names)

    def build_encoded_names(self, column: str) -> list[str]:
        if not self.dims:
            return [column]
        return [f"{column}_{self.name}_{i}" for i in range(self.size)]

    def encode(self, mat: NDArray, coords: DataFrame) -> DataFrame:
        """Encode the data into the space grid.

        Parameters
        ----------
        data : DataFrame
            Dataframe that contains the coordinate information and the column
            to encode.
        column : str, optional
            Column name to encode. Default is "intercept".

        Returns
        -------
        DataFrame
            Encoded dataframe.

        """
        vals = mat.ravel()
        nrow, ncol = mat.shape

        row = np.repeat(np.arange(nrow, dtype=int), ncol)
        col = np.zeros(nrow, dtype=int)
        if self.dims:
            col = (
                coords.merge(
                    self.span.reset_index(), how="left", on=self.dim_names
                )
                .eval("index")
                .to_numpy()
            )
        col = np.add.outer(ncol * col, np.arange(ncol))
        return coo_matrix((vals, (row, col)), shape=(nrow, self.size * ncol))

    def build_smoothing_prior(
        self,
        lam: float | dict[str, float] = 0.0,
        lam_mean: float = 0.0,
        scale_by_distance: bool = False,
    ) -> dict[str, NDArray]:
        """Create the smoothing Gaussian prior for the space.

        Parameters
        ----------
        lam : float or dict of float, optional
            Smoothing parameter for each dimension. Default is 0.0.
        lam_mean : float, optional
            Smoothing parameter for the mean. Default is 0.0.
        scale_by_distance : bool, optional
            Whether to scale the smoothing parameter by the distance. Default
            is False.

        Returns
        -------
        dict of NDArray
            Dictionary containing the prior information.

        """
        if isinstance(lam, float):
            lam = {name: lam for name in self.dim_names}
        mat, sd = coo_matrix((0, self.size)), np.empty((0,))

        mats_default = list(map(identity, self.dim_sizes))
        sds_default = list(map(np.ones, self.dim_sizes))

        for i, dim in enumerate(self.dims):
            lam_i = lam[dim.name]
            if lam_i > 0.0 and isinstance(dim, NumericalDimension):
                mats = mats_default.copy()
                mats[i] = dim.build_smoothing_mat()
                mat = vstack([mat, functools.reduce(kron, mats)])

                sds = sds_default.copy()
                sds[i] = dim.build_smoothing_sd(lam_i, scale_by_distance)
                sd = np.hstack([sd, functools.reduce(_flatten_outer, sds)])

        if lam_mean > 0.0:
            mat = vstack([mat, np.repeat(1 / self.size, self.size)])
            sd = np.hstack([sd, [1 / np.sqrt(lam_mean)]])

        # TODO: regmod cannot recognize sparse array as prior, this shouldn't
        # be necessary in the future
        return dict(mat=mat, sd=sd)


def _flatten_outer(x: NDArray, y: NDArray) -> NDArray:
    """Flatten the outer product of two arrays.

    Parameters
    ----------
    x : NDArray
        First array.
    y : NDArray
        Second array.

    Returns
    -------
    NDArray
        Flattened outer product of the two arrays.

    """
    return np.outer(x.ravel(), y.ravel()).ravel()
