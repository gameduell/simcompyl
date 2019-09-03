
SimComPyl - composable, compiled and pure python
====
|Build Status| |Coverage| |Binder|
------------------------------------------------

A Framework for Discrete Time Simulations
----

SimComPyl is a framework to write discrete time simulation. It focuses on ease
of development as well as execution speed. This is archived by compiling pure
composable python classes with numba:

  Thousends of decisions of millions of individuals 
  can be simulated within seconds.

The framework can be used to run any time-descrete simulation, e.g. you could use
it to get more insights on the interactions and behaviours taking place in sea-life, 
a mesh network or on your website.


Feature Overview
----
- pure python for rapid development and debugging
- designed for composability, so you can develop and extend step by step
- numba_ compiled execution for blazing fast execution
- integration of pandas_ and holoviews_ for (visual) analysis
- extendable execution modes and tracing capabilities
- unit and integration tests with high coverage

.. _numba: http://numba.pydata.org
.. _pandas: http://pandas.pydata.org
.. _holoviews: http://holoviews.org

.. |Build Status| image:: https://travis-ci.org/gameduell/simcompyl.svg?branch=master
   :target: https://travis-ci.org/gameduell/simcompyl
.. |Coverage| image:: https://coveralls.io/repos/github/gameduell/simcompyl/badge.svg?branch=master 
   :target: https://coveralls.io/github/gameduell/simcompyl?branch=master
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/gameduell/simcompyl/master
