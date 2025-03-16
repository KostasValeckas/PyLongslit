.. _extract_1d:

Extracting 1D spectrum
======================

This procedure extracts the 1D spectrum from the 2D spectrum (counts vs. spectral pixel).

There are 2 ways in the software to extract the 1D spectrum:

1. **Optimal extraction**: This is the default method.
2. **Summing the counts**: This is done by summing the counts along the spatial axis.

Both of these are described below.
For details on error estimation, please see the :ref:`note on uncertainties <uncertainties>`.

.. _optimal:

Optimal extraction
-------------------

The routine is called by the command:

.. code-block:: bash

    pylongslit_extract_1d PATH_TO_CONFIG_FILE

The routine will extract the 1D spectrum from the 2D spectrum using the optimal extraction method 
described in `Horne (1986) <https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract>`_.

The optimal extraction algorithm uses the profiles fitted from the :ref:`object tracing <objtrace>` routine 
to estimate the profile of the object in the spatial direction. The algorithm then weights the counts in the
2D spectrum according to the profile and sums the counts to get the 1D spectrum: 

.. math::

    S(p_{\lambda}) = \frac{\sum_{i} P_i(p_{\lambda}) \cdot D_i(p_{\lambda}) / \sigma_i(p_{\lambda})^2}
                {\sum_{i} P_i^2(p_{\lambda}) / \sigma_i(p_{\lambda})^2}

Where:

- :math:`S(p_{\lambda})` is the extracted 1D spectrum as a function of spectral pixel :math:`p_{\lambda}`.
- :math:`P_i(p_{\lambda})` is the spatial profile of the object at spatial pixel :math:`i` for spectral pixel :math:`p_{\lambda}`.
- :math:`D_i(p_{\lambda})` is the observed data (counts) at spatial pixel :math:`i` for spectral pixel :math:`p_{\lambda}`.
- :math:`\sigma_i(p_{\lambda})` is the uncertainty (noise) at spatial pixel :math:`i` for spectral pixel :math:`p_{\lambda}`.

The spectrum is then wavelength calibrated using the wavelength solution obtained from the :ref:`wavelength calibration <wavecalib>` routine.

Upon exiting, the routine will display the extracted 1D spectrum 
(taken from the SDSS_J213510+2728 example dataset):

.. image:: pictures/1dhorne.png
    :width: 100%
    :align: center

You can use the sliders to crop out any noise at the edges for better visualization.

The extracted 1D spectrum is saved in the output directory specified in the configuration file, 
with the filename ``"1d_science_FILENAME.dat"`` or ``"1d_standard_FILENAME.dat"``.
The files have three columns: wavelength, counts, and variance.

.. _sum:

Summing the counts
------------------

This extraction method is more simple than the optimal extraction method. 
Generally, you will get the more precise results with less noise using the optimal extraction method.
This method is however useful for edge cases where the object profile needs to be 
tightly constrained.

The routine is called by the command:

.. code-block:: bash

    pylongslit_extract_simple_1d PATH_TO_CONFIG_FILE

This procedure counts the number of counts in the region defined by the object 
center +/- the FWHM of the object.

Firstly, a QA is shown to display this region (from the SDSS_J213510+2728 example dataset):

.. image:: pictures/simple_region.png
    :width: 100%
    :align: center

Zoomed in:

.. image:: pictures/simple_zoom.png
    :width: 100%
    :align: center

If the region is not correct, you should revise the :ref:`object tracing <objtrace>` routine.

Then, the software will use ``"photutils.aperture.RectangularAperture"`` to sum the counts in the region.
You can read the docs for the method at `photutils <https://photutils.readthedocs.io/en/2.0.2/api/photutils.aperture.RectangularAperture.html#photutils.aperture.RectangularAperture.do_photometry>`_.

The spectra are then plotted and saved the same way as in the :ref:`optimal extraction <optimal>` routine.