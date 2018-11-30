from setuptools import setup, find_packages

setup(
    name = "simulate",
    version = "0.1",
    packages = find_packages(),
    scripts = [],
    package_data = {
        '': ['*.txt', '*.rst'],
    },

    # metadata for upload to PyPI
    author = "wabu",
    author_email = "wabu@fooserv.net",
    description = "Simulation Framework - fast and composabe",
    license = "MIT",
    keywords = "simulate monte-carlo numba population",
    url = "https://github.com/wabu/pyadds",
)
