from setuptools import setup, find_packages

setup(
    name="natten-rocm",
    version="0.21.0",
    description="Drop-in replacement for NATTEN on AMD ROCm (gfx1030 / RDNA2) using PyTorch FlexAttention",
    author="natten-flex-rocm contributors",
    packages=find_packages(include=["natten*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
