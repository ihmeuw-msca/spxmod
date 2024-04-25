import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src" / "spmat"

    sys.path.insert(0, src_dir.as_posix())

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = ["regmod==0.1.2", "Cython"]

    test_requirements = [
        "pytest",
        "pytest-mock",
    ]

    doc_requirements = []

    setup(
        name="spxmod",
        version="0.3.0",
        description="Coefficient-smoothing regression models",
        long_description=long_description,
        license="LICENSE",
        url="https://github.com/ihmeuw-msca/spxmod",
        author="IHME Math Sciences",
        author_email="ihme.math.sciences@gmail.com",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "dev": doc_requirements + test_requirements,
        },
        ext_modules=cythonize("src/spxmod/linalg.pyx"),
        include_dirs=[numpy.get_include()],
        zip_safe=False,
    )
