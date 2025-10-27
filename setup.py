from setuptools import setup, find_packages

setup(
    name="rdt-kernel",
    version="1.0.0",
    author="Steven Reid",
    author_email="sreid1118@gmail.com",
    description="Recursive Diffusion-Type Kernel — an entropy-regulated diffusion operator implemented in PyTorch.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RRG314/rdt-kernel",
    project_urls={
        "Documentation": "https://pypi.org/project/rdt-kernel",
        "Source": "https://github.com/RRG314/rdt-kernel",
        "Tracker": "https://github.com/RRG314/rdt-kernel/issues",
    },
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="diffusion, pde, entropy, torch, physics, kernel, RDT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=["torch>=1.12"],
    include_package_data=True,
)
