from setuptools import setup, find_packages

setup(
    name="CellAgentChat",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.22.3",
        'pandas >= 1.5.0',
        'scanpy >= 1.9.1',
        # We should not go beyond Mesa 1.0.0 for API stability
        'Mesa == 1.0.0',
        # It is advised not to put torch as a requirement in setup.py
        # As torch can be installed via conda instead
        #'torch >= 1.12.1',
        'scipy >= 1.9.1',
        'seaborn >= 0.12.0',
        'matplotlib >= 3.6.0',
        'pyslingshot >= 0.0.2',
        'anndata >= 0.8.0',
        'scanpy >= 1.9.6'
    ],
    author='Vishvak Raghavan',
    author_email='vishvak.raghavan@mail.mcgill.ca',
    description='Harnessing Agent-Based Modeling in CellAgentChat to Unravel Cell-Cell Interactions from Single-Cell Data',
    url='https://github.com/mcgilldinglab/CellAgentChat',
)
