from setuptools import setup, find_packages

setup(
    name="gated_deltanet",
    version="0.1.0",
    description="GatedDeltaNet implementation with varlen support",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.8.0",
        "causal-conv1d",
        "flash-linear-attention",
    ],
    extras_require={
        "test": ["pytest>=7.0", "pytest-cov"],
    },
)
