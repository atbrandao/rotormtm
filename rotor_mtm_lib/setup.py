import os
import re
import sys
import io

from setuptools import setup, find_packages

def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()

def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = version("rotor_mtm/__init__.py")

REQUIRES = [
    "numpy",
    "pint",
    "scipy",
    "plotly",
    "ross-rotordynamics"
]

setup(
    name="rotor_mtm",
    version=VERSION,
    description="Complemento da biblioteca ROSS para modelagem de metaestruturas giroscópicas.",
    author="André Brandão",
    author_email="andrebrandao@petrobras.com.br",
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRES,
)
