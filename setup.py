from setuptools import setup, find_packages


def _long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="arbitrary-woodland",
    version="0.0.1",
    long_description=_long_description(),
    packages=find_packages(),
)
