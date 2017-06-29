# mapper
[![Build Status](https://travis-ci.org/matthewgilbert/mapper.svg?branch=master)](https://travis-ci.org/matthewgilbert/mapper)
[![Coverage Status](https://coveralls.io/repos/github/matthewgilbert/mapper/badge.svg?branch=master)](https://coveralls.io/github/matthewgilbert/mapper?branch=master)

# Description

`mapper` provides functionality for mapping to and from generic exposures and
tradeable instruments for financial assets.

An example of this might be on 2016-12-01 we would have `CL1 -> CLZ16`, i.e.
the first generic for Crude oil on the above date corresponds to trading the
December 2016 contract.

The main features of `mapper` include:

- creating continuous return series for Futures instruments
- creating time series of percentage allocations to tradeable contracts
- creating instrument trade lists

The design layout of `mapper` for mapping from generics to tradeable
instruments is as follows

![workflow](/mapper.png)

The layout for mapping from tradeables to generics is as follows

![workflow](/mapper2.png)

# Install

You can pip install this package from github, i.e.

```
pip install git+git://github.com/matthewgilbert/mapper.git@master
```
