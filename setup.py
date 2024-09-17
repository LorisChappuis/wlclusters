from setuptools import setup

setup(
    name='wlclusters',
    version='0.1.0',
    packages=['wlclusters'],
    install_requires=[
        'numpy',
        'pymc',
        'astropy',
        'scipy',
        'tqdm'
    ],
    author='Loris Chappuis',
    description='A package to extract 1D shear profiles around galaxy cluster and fit them via density profile forward modeling',

)