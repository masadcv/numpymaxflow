import glob
import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


# get numpy as dependency when it is not pre-installed
# from: https://stackoverflow.com/a/54128391/798093
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        import builtins

        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


def get_extensions():
    ext_dir = "numpymaxflow"
    include_dirs = [ext_dir]

    sources = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
    # print(sources)
    define_macros = []
    extra_link_args = []
    extra_compile_args = []

    # add any compile flags here
    # compile release
    extra_compile_args += ["-g0"]

    if not sources:
        return []  # compile nothing

    ext_modules = [
        Extension(
            name="numpymaxflow",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    install_requires = fp.read().splitlines()

setup(
    name="numpymaxflow",
    version="0.0.2",
    description="numpymaxflow: Max-flow/Min-cut in Numpy for 2D images and 3D volumes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masadcv/numpymaxflow",
    author="Muhammad Asad",
    author_email="masadcv@gmail.com",
    license="BSD-3-Clause License",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    cmdclass={"build_ext": build_ext},  # .with_options(no_python_abi_suffix=True)},
    packages=find_packages(exclude=("data", "docs", "examples", "scripts", "tests")),
    ext_modules=get_extensions(),
    python_requires=">=3.6",
)
