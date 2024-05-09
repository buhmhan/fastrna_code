#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# Basic library
from setuptools import Extension, setup
from Cython.Build import cythonize

# Additional includes
import numpy as np

# Compile flags
c_flags = ["-O3"]
link_flags = []

# Main script
extensions = [
    Extension(
        "fastrna.core",
        ["fastrna/core.pyx"],
        include_dirs=[
            np.get_include(),
        ],
        extra_compile_args=c_flags,
        extra_link_args=link_flags,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "fastrna.eigen_funcs",
        ["fastrna/eigen_funcs.pyx"],
        include_dirs=[
            np.get_include(),
        ],
        extra_compile_args=c_flags,
        extra_link_args=link_flags,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "fastrna.utils",
        ["fastrna/utils.pyx"],
        include_dirs=[
            np.get_include(),
        ],
        extra_compile_args=c_flags,
        extra_link_args=link_flags,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "fastrna.de",
        ["fastrna/de.pyx"],
        include_dirs=[
            np.get_include(),
        ],
        extra_compile_args=c_flags,
        extra_link_args=link_flags,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    name='fastrna',
    version='0.1.0',
    description='Fast RNA processing library',
    ext_modules=cythonize(extensions),
    zip_safe=False,
    install_requires=[
        'numpy>=1.15.4',  # Specify minimum version
        'scipy>=1.1.0',  # Specify minimum version
        'pandas',  # need for DE
        'statsmodels',  # need for DE
        'pylbfgs',  # need for DE
    ])
