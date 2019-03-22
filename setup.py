from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # Library description
    name="olfactory",
    version="0.1.0",
    description="common tools for processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="utils tools processing analysis",

    # Packages / Modules
    packages=find_packages(),
    install_requires=[],

    # Code source and license
    url="https://github.com/OctaveLauby/olfactory",
    author="Octave Lauby",
    author_email="",
    license="Apache 2.0",

    # More
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
