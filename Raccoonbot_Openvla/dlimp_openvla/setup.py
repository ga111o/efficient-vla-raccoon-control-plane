from setuptools import find_packages, setup

setup(
    name="dlimp",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # NOTE: tensorflow was pinned to ==2.13.1, which has no Python 3.12 wheel
        #       and blocked `pip install -e .` on py3.12. Loosened to match openvla.
        "tensorflow>=2.15",
        "tensorflow_datasets>=4.9.2",
    ],
    extras_require={
        "convert": [
            "tqdm",
            "tqdm-multiprocess==0.0.11",
        ],
        "dev": [
            "pre-commit==3.3.3",
        ],
    },
)
