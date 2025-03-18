.. _setup:


Setup file
====================

This is not a pipeline step, but an introduction to a setup file 
(`setup.py`), that all of the following pipeline steps will read
in order to fetch relevant libraries, methods, data, constants etc.

The `setup.py` script will also read in the products from :ref:`pre-processing <pre_processing>`:
`arcsub.fits`, `arcsub_std.fits`, `obj.fits` and `std.fits`.

The parameters that can be set are documentet in the `setup.py` file.
We advise to keep the default values, and only change them if you are 
getting bad results in the pipeline steps.

.. note:: 
    The procedure of setting up python libraries and constants by direct
    execution of a script at every step is not recommended, as it is not a good
    practice with regards to performance, security and maintainability. 
    For now this is something that works, but we do not recommend this coding
    practise in general, and we to tend to move away from this in the future.