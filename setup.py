from setuptools import setup, find_packages

setup(
    name = "simcompyl",
    description = "Simulations, composable, compiled, pure python",
    version = "0.1",

    # metadata for upload to PyPI
    author = "wabu",
    author_email = "wabu@fooserv.net",
    long_description=open('README.md').read(),
    install_requires=list(open('requirements.txt').read().strip().split('\n')),
    setup_requires=["pytest-runner"],
    tests_require=['pytest'],
    packages = find_packages(),
    scripts = [],
    package_data = {},
    license = "MIT",
)
