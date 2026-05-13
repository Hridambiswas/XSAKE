from setuptools import setup, find_packages

setup(
    name="xsake",
    version="0.1.0",
    description="XLA-Fused Sparse Attention Kernel Engine with Head-Adaptive Dynamic Sparsity",
    author="Hridam Biswas",
    packages=find_packages(exclude=["tests*", "benchmarks*", "scripts*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.25",
        "jaxlib>=0.4.25",
        "flax>=0.8.0",
        "optax>=0.2.2",
        "orbax-checkpoint>=0.5.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "gpu": ["jax[cuda12]>=0.4.25"],
        "tpu": ["jax[tpu]>=0.4.25"],
        "training": [
            "datasets>=2.18.0",
            "tokenizers>=0.15.0",
            "transformers>=4.39.0",
            "tensorflow>=2.15.0",
        ],
        "observability": [
            "wandb>=0.16.0",
            "prometheus-client>=0.20.0",
            "fastapi>=0.110.0",
            "uvicorn>=0.29.0",
            "plotly>=5.20.0",
        ],
        "dev": ["pytest>=8.1.0", "pytest-xdist>=3.5.0"],
    },
)
