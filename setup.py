from setuptools import setup, find_packages

setup(
    name="sendepsic",
    version="0.1.0",
    description="Atomic structure analysis package with radial average & variance profiles for ePSIC",
    author="Jinseok Ryu",
    author_email="jinseuk56@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0",
        "scipy",
        "matplotlib",
        "py4DSTEM",
        "hyperspy",
        "scikit-learn",
        "pandas",
        "tifffile",
        "ipywidgets",
        "shapely",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
