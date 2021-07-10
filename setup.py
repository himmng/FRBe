import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="FRBe",
    version='0.0.1',
    author='Himanshu Tiwari',
    author_email="himanshuhimang@gmail.com",
    packages=setuptools.find_packages(),
    package_data={'': ['']},
    install_requires=["numpy", "scipy", "matplotlib", "emcee",],
    url="https://github.com/himmng/FRBe",
    license="MIT License",
    description="Fast Radio Bursts Estimator",
    include_package_data=True,
    keywords=["FRBe",
              "Fast Radio Bursts",
              "Radio Bursts",
              "cosmology",
              "MCMC"],
    project_urls={
        "Documentation": "coming soon!"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
