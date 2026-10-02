from importlib.metadata import version

from spxmod.dimension import (
    CategoricalDimension,
    Dimension,
    NumericalDimension,
    build_dimension,
)
from spxmod.model import XModel
from spxmod.space import Space
from spxmod.variable_builder import VariableBuilder

__version__ = version("spxmod")

__all__ = [
    "CategoricalDimension",
    "Dimension",
    "NumericalDimension",
    "Space",
    "VariableBuilder",
    "XModel",
    "__version__",
    "build_dimension",
]
