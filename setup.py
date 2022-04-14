from os import path

from setuptools import setup, find_packages

BASE_PATH = path.dirname(path.abspath(__file__))

with open(f"{BASE_PATH}/requirements.txt", "r") as fp:
    requirements = fp.read().splitlines()

setup(
    name="hades",
    packages=find_packages(include=["hades", "hades.*"]),
    install_requires=requirements,
)
