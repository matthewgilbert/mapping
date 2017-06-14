from setuptools import setup
import re

# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package#7071358
VERSIONFILE = "mapper/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." %
                       (VERSIONFILE,))

LONG_DESCRIPTION = """
`mapper` provides functionality for mapping to and from generic exposures and
tradeable instruments for financial assets.

An example of this might be on 2016-12-01 we would have `CL1 -> CLZ16`, i.e.
the first generic for Crude oil on the above date corresponds to trading the
December 2016 contract.

The main features of `mapper` include:

- creating continuous return series for Futures instruments
- creating time series of percentage allocations to tradeable contracts
- creating instrument trade lists
"""

setup(name='mapper',
      version=verstr,
      description='Mappings for generic and tradeable futures instruments',
      long_description=LONG_DESCRIPTION,
      url='https://github.com/MatthewGilbert/mapper',
      author='Matthew Gilbert',
      author_email='matthew.gilbert12@gmail.com',
      license='MIT',
      platforms='any',
      install_requires=['pandas', 'numpy', 'cvxpy'],
      packages=['mapper', 'mapper.tests'],
      test_suite='mapper.tests',
      zip_safe=False)
