# mapping
[![Build Status](https://travis-ci.org/matthewgilbert/mapping.svg?branch=master)](https://travis-ci.org/matthewgilbert/mapping)
[![Coverage Status](https://coveralls.io/repos/github/matthewgilbert/mapping/badge.svg?branch=master)](https://coveralls.io/github/matthewgilbert/mapping?branch=master)

# Description

`mapping` provides functionality for mapping to and from generic exposures and
tradeable instruments for financial assets.

An example of this might be on 2016-12-01 we would have `CL1 -> CLZ16`, i.e.
the first generic for Crude oil on the above date corresponds to trading the
December 2016 contract.

The main features of `mapping` include:

- creating continuous return series for Futures instruments
- creating time series of percentage allocations to tradeable contracts
- creating instrument trade lists

## Details

The mapping of instruments to and from generics is equivalent to solving the
equation `Ax = b` where `A` is the weights and `b` is the instrument holdings.
When `Ax = b` has no solution we solve for `x'` such that `Ax'` is closest to
`b` in the least squares sense with the additional constraint that
`sum(x') = sum(instruments)`.

A more realistic example of a mapping is given below

<table>
  <tr>
    <td></td>
    <th colspan="2">Generic</th>
    <th colspan="3" align="center">Instruments</th>
  </tr>
  <tr>
    <td></td>
    <td>CL1</td>
    <td>CL2</td>
    <td>Scenario 1</td>
    <td>Scenario 2</td>
    <td>Scenario 3</td>
  </tr>
  <tr>
    <td>CLX16</td>
    <td>0.5</td>
    <td>0</td>
    <td>10</td>
    <td>10</td>
    <td>10</td>
  </tr>
  <tr>
    <td>CLZ16</td>
    <td>0.5</td>
    <td>0.5</td>
    <td>20</td>
    <td>20</td>
    <td>25</td>
  </tr>
  <tr>
    <td>CLF17</td>
    <td>0</td>
    <td>0.5</td>
    <td>10</td>
    <td>11</td>
    <td>11</td>
  </tr>
</table>

Which would result in the following solutions to the mapping from instruments
to generics

<table>
  <tr>
    <td>Generic</td>
    <td>Scenario 1</td>
    <td>Scenario 2</td>
    <td>Scenario 3</td>
  </tr>
  <tr>
    <td>CL1</td>
    <td>20</td>
    <td>19.5</td>
    <td>22</td>
  </tr>
  <tr>
    <td>CL2</td>
    <td>20</td>
    <td>21.5</td>
    <td>24</td>
  </tr>
</table>

## Usage

A general workflow for using `mapping` for mapping from generics to tradeable
instruments is as follows

![workflow](/mapper.png)

The layout for mapping from tradeables to generics is as follows

![workflow](/mapper2.png)

# Install

You can pip install this package from github, i.e.

```
pip install git+git://github.com/matthewgilbert/mapping.git@master
```
