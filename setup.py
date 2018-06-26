from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import os
import glob
import numpy as np

base_path = ["arbitrary_woodland"]

source_files = [*glob.glob(os.path.join(*base_path, "*.pyx"))]

include_dirs = [os.path.join(*base_path), np.get_include()]

extensions = [
    Extension(
        name="arbitrary_woodland._tree",
        sources=source_files,
        language="c",
        include_dirs=include_dirs,
    )
]


def _long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="arbitrary-woodland",
    version="0.0.1",
    long_description=_long_description(),
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)
