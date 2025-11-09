from setuptools import setup, find_packages

setup(
    name="k_personal_projects",        # Replace with your project/package name
    version="0.1.0",
    packages=find_packages(),        # Automatically finds packages in your repo
    install_requires=[               # List runtime dependencies
        # "numpy>=1.24",
        # "pandas>=2.0",
    ],
    python_requires=">=3.8",
    author="KC",
    url="https://github.com/kenchan323/k_personal_projects",  # optional
)