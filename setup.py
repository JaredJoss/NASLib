import os
from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))

version_path = os.path.join(cwd, 'naslib', '__version__.py')
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())
setup(
    name='naslib',
    version=version,
    description='NASLib: A modular and extensible Neural Architecture Search (NAS) library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='AutoML Freiburg',
    author_email='zelaa@cs.uni-freiburg.de',
    url='https://github.com/automl/NASLib',
    license='Apache License 2.0',
    classifiers=['Development Status :: 1 - Beta'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    platforms=['Linux'],
    install_requires=requirements,
    keywords=['NAS', 'automl'],
    test_suite='pytest'
)
