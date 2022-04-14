from setuptools import setup, find_packages

setup(
    name='iss_analysis',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Petr Znamenskiy',
    author_email='petr.znamenskiy@crick.ac.uk',
    description='Analysis of in situ sequencing data',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm'
    ]
)
