from setuptools import setup, find_packages


setup(
    name="iss_analysis",
    version="0.1",
    packages=find_packages(),
    url="",
    license="MIT",
    author="Petr Znamenskiy",
    author_email="petr.znamenskiy@crick.ac.uk",
    description="Analysis of in situ sequencing data",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pciSeq",
        "h5py",
        "defopt",
        "bg-atlasapi",
        "anndata", 
        "abc_atlas_access @ git+https://github.com/alleninstitute/abc_atlas_access@main",  # The allen brain atlas cache
        "iss_preprocess @ git+ssh://git@github.com/znamlab/iss-preprocess.git"
    ],
    entry_points={
        "console_scripts": ["pick_genes = iss_analysis.pick_genes:entry_point"]
    }
)

