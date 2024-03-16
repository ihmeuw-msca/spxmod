import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csc_matrix


class Dimension:
    """Dimension used for grouped variable smoothing.

    Parameters
    ----------
    name : str or list of str
        Name of the dimension column in the data. When it is a list of str, it
        is used for multi-indexing.
    type : str or list of str
        Dimension type. When it is a list of str, it is used for multi-indexing.

    """

    def __init__(
        self,
        name: str | list[str],
        type: str | list[str],
        label: str | None = None,
    ) -> None:
        self.name = [name] if isinstance(name, str) else list(name)
        self.type = [type] if isinstance(type, str) else list(type)
        self.label = "*".join(self.name) if label is None else str(label)

        self._vals = None

    @property
    def vals(self) -> list[int | float]:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return self._vals

    @property
    def size(self) -> int:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return len(self._vals)

    def set_vals(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        combinations = list(
            itertools.product(*[np.unique(data[name]) for name in self.name])
        )
        self._vals = pd.DataFrame(combinations, columns=self.name).reset_index()

    def get_dummy_names(self, column: str) -> list[str]:
        """Get the dummy variable names for the dimension.

        Parameters
        ----------
        column : str
            Column name to use for the dummy variable names.

        Returns
        -------
        list of str
            Dummy variable names for the dimension.

        """
        return [f"{column}_{self.label}_{i}" for i in range(self.size)]

    def get_dummies(self, data: DataFrame, column: str = "intercept") -> DataFrame:
        """Get the dummy variables for the dimension.

        Parameters
        ----------
        data : DataFrame
            Data to get the dummy variables from.
        column : str, default "intercept"
            Column to use for the dummy variable values. Default is "intercept".

        Returns
        -------
        DataFrame
            Dummy variables data frame for the dimension.

        """
        value = data[column].to_numpy()

        row_index = np.arange(len(data))
        col_index = (
            data[self.name]
            .merge(self._vals, how="left", on=self.name)["index"]
            .to_numpy()
        )

        mat = csc_matrix((value, (row_index, col_index)), shape=(len(data), self.size))
        columns = self.get_dummy_names(column)
        return pd.DataFrame.sparse.from_spmatrix(mat, index=data.index, columns=columns)
