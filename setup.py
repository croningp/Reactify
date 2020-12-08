from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="Reactify",
    version="0.1",
    description="Reactivity detection using deep neural nets",
    url="https://github.com/croningp/Reactify",
    author="Hessam Mehr, Dario Caramelli",
    author_email="Hessam.Mehr@glasgow.ac.uk; Dario.Caramelli@glasgow.ac.uk",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(),
)
