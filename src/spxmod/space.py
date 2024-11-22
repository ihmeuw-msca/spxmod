from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.sparse import coo_matrix, diags, hstack, identity, kron, vstack

from spxmod.dimension import Dimension, NumericalDimension, build_dimension
from spxmod.typing import DataFrame, NDArray, Series


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

    def encode_coords(self, coords: DataFrame) -> DataFrame:
        weights = functools.reduce(
            lambda x, y: x.merge(y, on="row", how="outer"),
            (dim.encode_coords(coords) for dim in self.dims),
        )
        dim_sizes = [dim.size for dim in self.dims]
        dim_names = [dim.name for dim in self.dims]
        res_sizes = np.hstack([1, np.cumprod(dim_sizes[::-1][:-1], dtype=int)])[
            ::-1
        ]

        weights["col"] = 0
        weights["val"] = 1.0
        for dim_name, res_size in zip(dim_names, res_sizes):
            weights["col"] += weights[f"{dim_name}_col"] * res_size
            weights["val"] *= weights[f"{dim_name}_val"]
            weights.drop(
                columns=[f"{dim_name}_col", f"{dim_name}_val"], inplace=True
            )
        return weights

    def normalize_weights(
        self, weights: DataFrame, density: Series | None = None
    ) -> DataFrame:
        if density is not None:
            if not isinstance(density, Series):
                raise TypeError(
                    "density must be a pandas Series with index coincide with "
                    "the space dimensions."
                )
            density = density.rename("density").reset_index()
            missing_cols = set(self.span.columns) - set(density.columns)
            if missing_cols:
                raise ValueError(
                    f"Please provide {missing_cols} as the density index."
                )
            matched_density = self.span.merge(density, how="left")
            if matched_density["density"].isna().any():
                raise ValueError(
                    "Missing density value for certain kernel dimension."
                )
            density = matched_density["density"].to_numpy()
            weights["val"] *= density[weights["col"].to_numpy()]
        weights["val"] /= weights.groupby("row")["val"].transform("sum")
        return weights

    def encode(
        self, mat: NDArray, coords: DataFrame, density: Series | None = None
    ) -> coo_matrix:
        """Encode the data into the space grid.

        Parameters
        ----------
        mat
            Design matrix to be encoded, it should have the same number of rows
            with `coords`.
        coords
            Coordinate data frame for each row of the design matrix. It's
            columns should match with the space dimension name.

        Returns
        -------
        coo_matrix
            Encoded design matrix.

        """
        weights = self.encode_coords(coords)
        weights = self.normalize_weights(weights, density)
        row, col, val = weights[["row", "col", "val"]].to_numpy().T
        weights = coo_matrix((val, (row, col)), shape=(len(coords), self.size))
        return hstack([diags(cov) @ weights for cov in mat.T], format="coo")

    def build_smoothing_prior(
        self,
        size: int,
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
        mat, sd = coo_matrix((0, self.size * size)), np.empty((0,))

        mats_default = list(map(identity, self.dim_sizes))
        sds_default = list(map(np.ones, self.dim_sizes))

        for i, dim in enumerate(self.dims):
            lam_i = lam[dim.name]
            if lam_i > 0.0 and isinstance(dim, NumericalDimension):
                mats = mats_default.copy()
                mats[i] = kron(dim.build_smoothing_mat(), identity(size))
                mat = vstack([mat, functools.reduce(kron, mats)])

                sds = sds_default.copy()
                sds[i] = np.repeat(
                    dim.build_smoothing_sd(lam_i, scale_by_distance), size
                )
                sd = np.hstack([sd, functools.reduce(_flatten_outer, sds)])

        if lam_mean > 0.0:
            mat = vstack(
                [
                    mat,
                    kron(np.repeat(1 / self.size, self.size), identity(size)),
                ]
            )
            sd = np.hstack([sd, np.repeat(1 / np.sqrt(lam_mean), size)])

        # TODO: regmod cannot recognize sparse array as prior, this shouldn't
        # be necessary in the future
        return dict(mat=mat, sd=sd)

    def build_order_prior(
        self,
        order_dim: str = "",
        order: list[list[int]] | None = None,
    ) -> dict[str, NDArray]:
        mat = coo_matrix((0, self.size))
        if order_dim == "":
            return dict(mat=mat)

        mats_default = list(map(identity, self.dim_sizes))

        for i, dim in enumerate(self.dims):
            if dim.name == order_dim and isinstance(dim, NumericalDimension):
                mats = mats_default.copy()
                mats[i] = vstack(
                    [dim.build_order_mat(order_item) for order_item in order]
                )
                mat = vstack([mat, functools.reduce(kron, mats)])

        return dict(mat=mat)


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
