import glob
from setuptools import setup, find_packages


with open('README') as f:
    long_description = ''.join(f.readlines())


setup(
    name='LSSVMlib',
    version='2020.10.21',
    description='Python LS-SVM class for regression conforming to the scikit-learn API.',
    author='Danny E.P. Vanpoucke',
    license='MIT',
    keywords='LS-SVM, Least Squares Support Vector Machine, Regression, scikit-learn, machine learning, artificial intelligence',
    url='https://github.com/DannyVanpoucke/LSSVMlib',
    packages=find_packages(),
    install_requires=[
        'sklearn>=0.21.2',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'scipy>=1.2.1',
    ],
)
