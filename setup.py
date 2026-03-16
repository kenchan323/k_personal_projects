from setuptools import setup, find_packages

setup(
    name='k_personal_projects',
    version='0.1.0',
    package_dir={'': 'python'},
    packages=find_packages(where='python'),
    package_data={
        'kutils': ['data/etl/config/*.yaml'],
    },
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
        'arcticdb',
        'pyyaml',
        'scipy',
    ],
)
