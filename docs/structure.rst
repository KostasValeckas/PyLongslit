.. _structure:

Overview of all pipeline procedures and their products
======================================================

This page provides an overview of all the procedures that are part of the
PyLongslit pipeline, along with the products generated at each step. Click 
on any of the procedure names to get more information on that specific step.

All the procedures are given in the order they are intended to be called in.

Detector calibrations
----------------------

The following procedures produce an array of master frames, and lastly use 
them to produce reduced and wavelength-calibrated 2d spectra of the science and/or the standard star
observations.

For these procedures, the pipeline is designed to overwrite any products in place, so a repeated call 
to a procedure will overwrite a previous product placed in the output directory given in the :ref:`configuration file <conf>`.

Procedures for producing master calibration frames are:

.. list-table:: Procedures for producing master calibration frames
    :header-rows: 1

    * - Procedure
      - Products
    * - :ref:`Bias <bias>`
      - ``master_bias.fits``
    * - :ref:`Arc lamp combination <combine_arcs>`
      - ``master_arc.fits``
    * - :ref:`Identifying arc lines <identify>`
      - Arc line identification file (these are not placed in the output directory, but are dependent on :ref:`instrument and configuration <tested_instruments>`.)
    * - :ref:`Wavelength Calibration <wavecalib>`
      - ``good_lines.pkl``, ``reidentified_lines.pkl``, ``tilt_fit.pkl``, ``wavelen_fit.pkl``, ``wavelength_map.fits``
    * - :ref:`Flat-fielding <flat>`
      -  ``master_flat.fits`` 

After all the masters are produced, the :ref:`reduction procedure <reduce>` invokes them 
to produce reduced science and/or standard star frames. If :ref:`dark frames <dark>` are provided, these will also be invoked in this step. 
The output files are named ``"reduced_science_ORIGINAL_NAME.fits"`` or ``"reduced_standard_ORIGINAL_NAME.fits"``.

Further processing of reduced frames
-------------------------------------

In the following steps, the reduced frames are processed further. All these procedures
**alter the reduced frames in place** - if you wish to revert any changes, you have to
run the :ref:`reduction procedure <reduce>` again to reset the frames. Any of these procedures 
can in principle be skipped if not needed for your science case, but are generally recommended.
The reduced and wavelength calibrated 2d spectra can be inspected using the :ref:`2D spectrum viewer <2dspec>`.

1. :ref:`Cosmic Ray Removal <crr>`: Identify and remove cosmic ray hits from the reduced frames.
2. :ref:`A-B background subtraction <ab>`: Perform A-B background subtraction to remove sky background from the reduced frames.
3. :ref:`Cropping <crop>`: Crop the reduced frames in the spatial direction to improve sky-subtraction and object-tracing.
4. :ref:`Modelled sky subtraction <sky>`: Create a model of the sky background and subtract it from the reduced frames. This procedure will save the sky model with the name ``"skymap_REDUCED_FILENAME.fits"``.

1D-spectra extraction and flux calibration
-------------------------------------------

1. :ref:`Object tracing <objtrace>`: Trace and model the object spectrum. The procedure will produce files with the name  ``"obj_science_ORIGINAL_FILENAME.dat"`` or ``"obj_standard_ORIGINAL_FILENAME.dat"``.
2. :ref:`1D-spectra extraction <extract_1d>`: Extract the 1D-spectra in counts vs. wavelength. The spectra are saved as ``"1d_science_ORIGINAL_NAME.dat"`` or ``"1d_standard_ORIGINAL_NAME.dat"``.
3. :ref:`Sensitivity function <sensfunction>`: Calculate the response/sensitivity function for the detector. This is need for flux-calibration. The sensitivity function is stored as ``"sensfunc.dat"``.
4. :ref:`Flux-calibration <flux_calibrate>`: Applies the :ref:`sensitivity function <sensfunction>` to calibrate the spectra from counts vs. wavelength to :math:`\text{erg}/\text{cm}^2/\text{Å}/\text{s}` vs. wavelength. The files are saved as ``"1d_fluxed_science_ORIGINAL_FILENAME.dat"``.
5. :ref:`Combining of fluxed spectra <combine_spec>`: Combine the fluxed spectra from different exposures into a single 1D spectrum. The file is saved as defined by the user in the :ref:`configuration file <conf>`.

