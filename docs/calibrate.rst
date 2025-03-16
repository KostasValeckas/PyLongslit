.. _flux_calibrate:

Flux calibration
==================

This routine uses the :ref:`sensitivity function <sensfunction>`
to calibrate the :ref:`extracted 1D spectra <extract_1d>` to units 
of :math:`\text{erg}/\text{cm}^2/\text{Å}/\text{s}`.

Quick start
------------

The routine is called by the command:

.. code-block:: bash

    pylongslit_flux_calibrate PATH_TO_CONFIG_FILE


The algorithm for fluxing is as follows:

1. The counts in the spectrum are divided by the exposure time to get counts per second.

2. Multiply the counts per second with the :ref:`transmission_factor <transmission_factor>`.

3. Multiply the result with the :ref:`sensitivity function <sensfunction>`.
   This will give you the flux in units of :math:`\text{erg}/\text{cm}^2/\text{Å}/\text{s}`.

The routine will then display the flux calibrated spectrum 
(example from the GQ1218+0832 example dataset):

.. image:: pictures/flux_calib.png
    :width: 100%
    :align: center

The routine will only show the interval at which the :ref:`sensitivity function <sensfunction>`
is well behaved (these are the parts of the spectrum that were not cropped during 
:ref:`sensitivity function <sensfunction>` fitting). However, the whole 
spectrum is saved with the filename ``1d_fluxed_science_FILENAME.dat``. 
The file has three columns: wavelength, flux, and variance.

Parameter options
-----------------

You have to provide the path to the :ref:`extinction curve <transmission_factor>` as 
described in the chapter about the :ref:`sensitivity function <sensfunction>`.

Furthermore, these parameters have to be set in the configuration file:

.. code:: 

    "science" : {
        "exptime": 400,
        "airmass": 1.20
    }

- ``exptime``: The exposure time of the science frame in seconds.
- ``airmass``: The airmass of the observation.

You can not flux calibrate several frames with different exposure times or airmasses
with the same call to this routine - but for frames that have same exposure time and
(nearly) same airmass, you can flux calibrate them all at once.