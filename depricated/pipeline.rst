.. _pipeline:

Pipeline
========

This section describes the stepts needed to obtain a 
calibrated and reduced science spectrum, after the
:ref:`pre-processing <pre_processing>` steps have been completed. 
This means that at this point, you should have aquired the files
`arcsub.fits`, `arcsub_std.fits`, `obj.fits` and `std.fits`, and your 
overall file and directory structure should look like this (irrelevant files 
from :ref:`pre-processing <pre_processing>` have been omitted):

.. code-block:: bash

    ├── arcsub.fits
    ├── arcsub_std.fits
    ├── calibrate.py
    ├── database
    │   ├── lapalma.dat
    │   ├── map-g04-he-1.pdf
    │   ├── map-g04-he-2.pdf
    │   ├── map-g04-ne-1.pdf
    │   ├── map-g04-ne-2.pdf
    │   └── mylines_vac.dat
    ├── extract_1d.py
    ├── extract_science_1d.py
    ├── extract_std_1d.py
    ├── identify.py
    ├── mfeige110.dat
    ├── obj.fits
    ├── sensfunction.py
    ├── setup.py
    ├── standard.py
    └── std.fits

To obtain a 
calibrated and reduced science spectrum, follow these steps (in the order they are listed)
by clicking on the links below:

.. toctree:: 
    :maxdepth: 2
    
    setup
    identify
    extract_1d
    standard
    sensfunction
    calibrate

--------------------------------------------

.. toctree:: 

    pipeline_diagram


