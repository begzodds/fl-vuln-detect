from setuptools import setup, find_packages

setup(
    name="fl-vuln-detect",
    version="0.1.0",
    author="Begzod Abdumutalliev",
    author_email="bdmbzd99m27z259p@studenti.unime.it",
    description="Noise-robust federated learning for software vulnerability detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "flwr[simulation]>=1.8.0",
        "transformers>=4.40.0",
        "torch>=2.2.0",
        "datasets>=2.19.0",
        "omegaconf>=2.3.0",
        "scikit-learn>=1.4.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black>=24.0.0", "ruff>=0.4.0"],
    },
)
