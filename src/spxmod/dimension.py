import numpy as np
from msca.integrate.integration_weights import build_integration_weights
from scipy.sparse import coo_matrix

from spxmod.typing import DataFrame, NDArray


class Dimension:
    """Dimension used for grouped variable smoothing.

    Parameters
    ----------
    name : str
        Name of the dimension in the data.
    skipna : bool, optional
        Whether to exclude rows where name is nan. Default is True.

    """

    def __init__(
        self,
        name: str,
        interval: tuple[str, str] | None = None,
        skipna: bool = True,
    ) -> None:
        self.name = name
        self.interval = interval
        self.skipna = skipna
        self._grid: NDArray | None = None
        self._span: NDArray | None = None

    @property
    def span(self) -> NDArray:
        if self._span is None:
            raise ValueError("Dimension values are not set.")
        return self._span

    @property
    def grid(self) -> NDArray:
        if self._grid is None:
            raise ValueError("Dimension values are not set.")
        return self._grid

    @property
    def size(self) -> int:
        return len(self.span)

    def set_span(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        columns = [self.name] if self.interval is None else list(self.interval)
        if self.skipna:
            for column in columns:
                data = data.query(f"{column}.notna()")
        data = (
            data[columns]
            .drop_duplicates()
            .sort_values(columns, ignore_index=True)
        )

        if self.interval is None:
            self._grid = np.unique(data[self.name])
            self._span = self._grid.copy()
        else:
            lb, ub = self.interval
            grid = np.hstack([data.loc[0, lb], data[ub]])
            if not np.allclose(data[lb], grid[:-1]):
                raise ValueError("Range intervals contain gap(s)")
            self._grid = grid
            self._span = np.asarray(data.mean(axis=1))

    def encode_coords(self, coords: DataFrame) -> DataFrame:
        if self.interval is None:
            row = np.arange(len(coords), dtype=int)
            col = (
                coords[[self.name]]
                .merge(
                    DataFrame({self.name: self.span}).reset_index(),
                    on=self.name,
                    how="left",
                )["index"]
                .to_numpy()
            )
            val = np.ones(len(coords))
        else:
            val, (row, col) = build_integration_weights(
                coords[self.interval[0]].to_numpy(),
                coords[self.interval[1]].to_numpy(),
                self.grid,
                rule="midpoint",
            )
        return DataFrame(
            {
                "row": row,
                f"{self.name}_col": col,
                f"{self.name}_val": val,
            }
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class CategoricalDimension(Dimension):
    """Categorical dimension, mainly used for hierarchical relationships. All
    the values in the span do not have order. The common usage of this dimension
    is to emulate random effect in the linear mixed effect model.

    """

    def __init__(
        self,
        name: str,
        interval: tuple[str, str] | None = None,
        skipna: bool = True,
    ) -> None:
        if interval is not None:
            raise ValueError(
                "Interval is not supported for categorical dimension."
            )
        super().__init__(name, interval, skipna)


class NumericalDimension(Dimension):
    """Numerical dimension, the values in the space determine the ordered position
    of variable defined on this dimension. This class provide smoothing prior
    constructor used for introduce correlation between variables defined on this
    dimension.

    """

    def build_smoothing_mat(self) -> coo_matrix:
        """Create the smoothing prior matrix for the numerical dimension.

        Returns
        -------
        coo_matrix
            Smoothing prior matrix for the numerical dimension.

        """
        val = np.hstack([np.ones(self.size - 1), -np.ones(self.size - 1)])
        row = np.tile(np.arange(self.size - 1), 2)
        col = np.hstack([np.arange(self.size - 1), np.arange(1, self.size)])
        return coo_matrix((val, (row, col)), shape=(self.size - 1, self.size))

    def build_smoothing_sd(
        self, lam: float, scale_by_distance: bool
    ) -> NDArray:
        """Create the smoothing prior standard deviation vector for the numerical dimension.

        Parameters
        ----------
        lam : float
            Smoothing parameter.
        scale_by_distance : bool
            Whether to scale the prior vector by the distance between the dimension
            values.

        Returns
        -------
        NDArray
            Smoothing prior standard deviation vector for the numerical dimension.

        """
        sd = np.repeat(1 / np.sqrt(lam), self.size - 1)
        if scale_by_distance:
            delta = np.diff(self.span).astype(sd.dtype)
            delta /= delta.min()
            sd *= delta
        return sd

    def build_order_mat(self, order: list[list[int]]) -> coo_matrix:
        size = len(order)
        val = np.hstack([np.ones(size - 1), -np.ones(size - 1)])
        row = np.tile(np.arange(size - 1), 2)
        col = np.hstack([order[:-1], order[1:]])
        return coo_matrix((val, (row, col)), shape=(size - 1, self.size))


def build_dimension(name: str, dim_type: str, **kwargs) -> Dimension:
    """Create a dimension based on the dimension type.

    Parameters
    ----------
    name : str
        Name of the dimension in the data.
    dim_type : {'categorical', 'numerical'}
        Type of the dimension.
    skipna : bool, optional
        Whether to exclude rows where name is nan. Default is True.

    Returns
    -------
    Dimension
        Dimension based on the dimension type.

    """
    if dim_type == "categorical":
        return CategoricalDimension(name, **kwargs)
    elif dim_type == "numerical":
        return NumericalDimension(name, **kwargs)
    else:
        raise TypeError(f"Dimension type {dim_type} is not supported.")
