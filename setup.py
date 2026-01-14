from setuptools import setup, find_packages

setup(
    name="jesprit-project",
    version="0.1.0",
    description="Implementation of Joint ESPRIT Algorithm",
    
    # This tells setuptools "the root of our packages is in 'src'"
    package_dir={'': 'src'},
    
    # This automatically finds all packages
    # inside the 'src' directory.
    packages=find_packages(where='src'),
    
    # Specify dependencies
    install_requires=[
        'numpy','matplotlib'
    ],
)