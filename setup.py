from setuptools import setup, find_packages

setup(
    name = "simulate",
    version = "0.1",

    # metadata for upload to PyPI
    author = "wabu",
    author_email = "wabu@fooserv.net",
    description = "simulate behaviour of large populations blazing fast",
    long_description=open('README.rst').read(),
    install_requires=list(open('requirements.txt').read().strip().split('\n')),
    setup_requires=["pytest-runner"],
    tests_require=['pytest'],
    packages = ['simulate'],
    scripts = [],
    package_data = {},
    license = "MIT",
)
