import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimpv",
    version="v1.05",  
    author="Vincent M. Le Corre",
    description="optimPV: Optimization & Modeling tools for PV research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3",
    url="https://github.com/openPV-lab/optimPV",
    download_url="https://github.com/openPV-lab/optimPV/v1.05.tar.gz",
    packages=setuptools.find_packages(),
    keywords=[
        "Bayesian optimization",
        "Evolutionary optimization",
        "parameter extraction",
        "experimental design",
        "high throughput",
        "solar cells",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires=">=3.12",
    install_requires=[
        # Core
        "numpy>=1.2,<=2.0",
        "pandas>=1.4",
        "scipy>=1.0",
        "matplotlib>=3.5",
        "seaborn>=0.11",
        "openpyxl>=3.0",
        "pyodbc>=4.0",
        "gitpython>=3.1",

        # Jupyter
        "jupyterlab>=3.4",
        "jupyter",

        # Optimization / BO
        "scikit-optimize>=0.9",
        "emcee>=3.1",
        "pymoo>=0.6",
        "ax-platform>=1.2.1",
        "arviz>=0.15.1",
        "botorch>=0.6",
        "gpytorch",

        # ML
        "torch>=2.0",
        "torchvision>=0.15",
        "torchaudio>=2.0",

        # Domain-specific
        "pySIMsalabim>=1.3",
        "tk",
    ],
    extras_require={
        "dev": [
            "pytest",
            "twine",
            "black",
            "flake8",
            "coverage",
        ],
    },
    include_package_data=True,
)
