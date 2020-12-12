from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Reactify",
    version="0.1",
    description="General-purpose reactivity detection using proton NMR spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/croningp/Reactify",
    author="Hessam Mehr, Dario Caramelli",
    author_email="Hessam.Mehr@glasgow.ac.uk; Dario.Caramelli@glasgow.ac.uk",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(),
)
