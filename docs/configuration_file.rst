.. _conf:

Configuration files
===================

The configuration file is a ``.json`` file that contains all the parameters
that are needed to run the software. The file is divided into sections, 
each corresponding to a different step in the pipeline.

Every pipeline command takes the configuration file as input. From the 
configuration file the software knows where to look for the data, what to do with it,
and where to save the results. Every call has the following format:

.. code:: bash

    pylongslit_command PATH_TO_CONFIG_FILE

For example, if the command is the bias procedure ``pylongslit_bias``, and
the configuration file is located at ``/home/documents/SDSS_J213510+2728.json``,
the command would be:

.. code:: bash

    pylongslit_bias /home/documents/SDSS_J213510+2728.json

When first seeing the configuration file, the many parameters can be overwhelming, but most of them are 
constant for a given instrument. This means that once you have set them up
for a single successful run for your instrument, you can reuse most of the configuration file
for future runs.

**See the page on** :ref:`already tested instruments <tested_instruments>`
**for configuration files that have been tested**.

For a new instrument, you can
try to **reuse a configuration file from another instrument**, and change only
the parameters that are corrupting the run.

The software has a method for checking the configuration file for errors 
(it checks if the provided file paths exist, the raw data directories have files in them, some sanity checks 
on the parameters, etc.). It also displays the selected overscan region used for the :ref:`bias <bias>` subtraction. 
The method can be called by:

.. code:: bash

    pylongslit_check_config PATH_TO_CONFIG_FILE

It is not mandatory to use this method, but it is highly recommended before 
actually starting to process the data.

An example of a configuration file is shown below, followed by an 
explanation of every parameter. In the :ref:`tutorial <tutorial>`, even more
detailed explanation of the parameters is given.

.. code:: 

    {
        "instrument": {
            "name": "ALFOSC",
            "disperser": "Grating #4"
        },

        "detector": {
            "xsize": 500,
            "ysize": 2102,
            "dispersion": { 
                "spectral_dir": "y",
                "wavelength_grows_with_pixel": false
            },
            "gain": 0.16,
            "read_out_noise": 4.3,
            "overscan" : {
                "use_overscan": true,
                "overscan_x_start": 0,
                "overscan_x_end": 499,
                "overscan_y_start": 2064,
                "overscan_y_end": 2102
            }
        },

        "data": {
            "raw_data_hdu_index": 1
        },

        "bias": {
            "bias_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/bias",
            "bootstrap_errors": false
        },

        "flat": {
            "flat_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/flats",
            "bootstrap_errors": false,
            "skip_spatial": false,
            "knots_spectral_bspline": 70,
            "degree_spectral_bspline": 3,
            "knots_spatial_bspline": 4,
            "degree_spatial_bspline": 3,
            "R2_spatial_bspline": 0.4

        },

        "output": {
            "out_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/output"

        },

        "crr_removal" : {
            "science":{
                "frac": 0.3,
                "objlim": 6,
                "sigclip": 6.0,
                "niter": 2
            },
            "standard":{
                "frac": 0.3,
                "objlim": 4,
                "sigclip": 4.0,
                "niter": 3
            }
        },

        "background_sub" : {
            "pairs": {
                "1": {
                    "A": "ALHh080251.fits",
                    "B": "ALHh080252.fits"
                },
                "2": {
                    "A": "ALHh080252.fits",
                    "B": "ALHh080251.fits"
                }
            }
        },

        "science" : {
            "skip_science": false,
            "science_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/science",
            "exptime": 400,
            "airmass": 1.20
        },

        "standard" : {
            "skip_standard": false,
            "standard_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/standard",
            "exptime": 30,
            "airmass": 1.0421315680187,
            "starname": "BD332642",
            "flux_file_path": "/home/kostas/Documents/PyLongslit/database/bd33a.oke"

        },

        "arc" : {
            "arc_dir": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/arcs"
        },

        "combine_arcs" : {
            "skip_bias": false
        },

        "wavecalib" : {
            "offset_middle_cut": 0,
            "pixel_cut_extension": 2,
            "arcline_start": 0,
            "arcline_end": 500,
            "jump_tolerance": 0.05,
            "center_guess_pixtable": "/home/kostas/Documents/PyLongslit/database/alfosc_grating4_hene_pixtable.dat",
            "FWHM": 6,
            "TOL_MEAN": 2,
            "TOL_FWHM": 1,
            "REIDENTIFY_R2_TOL": 0.90,
            "ORDER_WAVELEN_1D": 5,
            "ORDER_SPECTRAL_TILT": 1,  
            "ORDER_SPATIAL_TILT": 4,
            "TILT_TRACE_R2_TOL": 0.99,
            "TILT_REJECT_LINE_FRACTION": 0.1,
            "SPATIAL_R2_TOL": 0.97,
            "reuse_reided_lines": false,
            "reuse_1d_sol": false,
            "reuse_line_traces": false,
            "reuse_2d_tilt_fit": false      
        },

        "sky" : {
            "sigma_cut": 3,
            "sigma_clip_iters": 5,
            "fit_order": 2
        },

        "trace" : {
            "object": {
                "spectral_pixel_extension": 10,
                "fwhm_guess": 2.5,
                "fwhm_thresh": 1,
                "center_thresh": 3,
                "SNR": 12,
                "fit_order_trace": 3,
                "fit_order_fwhm": 3,
                "fit_R2": 0.90,
                "use_bspline_obj": false,
                "use_bspline_fwhm": false,
                "knots_bspline": 4,
                "model": "Gaussian"
            },
            "standard": {
                "spectral_pixel_extension": 0,
                "fwhm_guess": 4,
                "fwhm_thresh": 4,
                "center_thresh": 3,
                "SNR": 70,
                "fit_order_trace": 2,
                "fit_order_fwhm": 2,
                "fit_R2": 0.99,
                "use_bspline_obj": true,
                "use_bspline_fwhm": true,
                "knots_bspline": 10,
                "model": "Gaussian"
            }
        },

        "obj_trace_clone" : {
            "archived_spec_root": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/output/obj_science_ALHh080251.dat",
            "frame_root": "/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/output/reduced_science_ALHh080252.fits"
        },

        "sensfunc": {
            "fit_order": 3,
            "use_bspline": true,
            "knots_bspline": 15
        },

        "flux_calib": {
            "path_extinction_curve": "/home/kostas/Documents/PyLongslit/database/lapalma.dat"
        },

        "combine": {
            "SDSS_J213510+2728": ["ALHh080251.fits", "ALHh080252.fits"]
        },

        "developer": {
            "debug_plots": true,
            "verbose_print": true

        }
    }

A brief explanation of every parameter is provided below. Please see the :ref:`tutorial <tutorial>` for more detailed explanation 
of every step and their relevant parameters. 

Instrument Configuration
------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``instrument.name``
      - string
      - The name of the instrument, simply for logging purposes.
    * - ``instrument.disperser``
      - string
      - The disperser used, simply for logging purposes.

Detector Configuration
----------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``detector.xsize``
      - integer
      - The number of pixels along the x axis.
    * - ``detector.ysize``
      - integer
      - The number of pixels along the y axis.
    * - ``detector.dispersion.spectral_dir``
      - string
      - The direction of the spectral axis in raw data, either "x" or "y".
    * - ``detector.dispersion.wavelength_grows_with_pixel``
      - boolean
      - true if the wavelength increases with pixel number for the spectral axis.
    * - ``detector.gain``
      - float
      - detector gain in electrons per count (ADU).
    * - ``detector.read_out_noise``
      - float
      - read-out noise in electrons.
    * - ``detector.overscan.use_overscan``
      - boolean
      - true if overscan is to be used, false otherwise (then the bias is estimated only from bias frames).
    * - ``detector.overscan.overscan_x_start``
      - integer
      - The starting pixel of the overscan region along the x axis.
    * - ``detector.overscan.overscan_x_end``
      - integer
      - The ending pixel of the overscan region along the x axis.
    * - ``detector.overscan.overscan_y_start``
      - integer
      - The starting pixel of the overscan region along the y axis.
    * - ``detector.overscan.overscan_y_end``
      - integer
      - The ending pixel of the overscan region along the y axis.

Data Configuration
------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``data.raw_data_hdu_index``
      - integer
      - Index of the HDU in the raw data FITS file that contains the data (usually 0 for single-extension FITS files, 1 for multi-extension FITS files).

Bias Configuration
------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``bias.bias_dir``
      - string
      - The directory where the bias frames are located (this directory should not contain any other files).
    * - ``bias.bootstrap_errors``
      - boolean
      - true if bootstrapping should be used to estimate the error in the bias frames, false otherwise (bootstrapping takes longer, but is more accurate).

Flat Field Configuration
------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``flat.flat_dir``
      - string
      - The directory where the flat-field frames are located (this directory should not contain any other files).
    * - ``flat.bootstrap_errors``
      - boolean
      - true if bootstrapping should be used to estimate the error in the flat-field frames, false otherwise (bootstrapping takes longer, but is more accurate).
    * - ``flat.skip_spacial``
      - boolean
      - true if slit-illumination correction should be skipped, false otherwise.
    * - ``flat.knots_spectral_bspline``
      - integer
      - The number of knots in the bspline when fitting the detector spectral response.
    * - ``flat.degree_spectral_bspline``
      - integer
      - The degree of the bspline when fitting the detector spectral response.
    * - ``flat.knots_spacial_bspline``
      - integer
      - The number of knots in the bspline when fitting the detector spatial response.
    * - ``flat.degree_spacial_bspline``
      - integer
      - The degree of the bspline when fitting the detector spatial response.
    * - ``flat.R2_spacial_bspline``
      - float
      - The rejection threshold for the bspline when fitting the detector spatial response.

Output Configuration
--------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``output.out_dir``
      - string
      - The directory where the output files should be saved.

Cosmic Ray Removal Configuration
--------------------------------
**Note:** The parameters below live under either ``crr_removal.science`` or ``crr_removal.standard``.
    
.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``crr_removal.<type>.frac``
      - float
      - The fraction of sigclip to use for the lower limit of the contrast detection.
    * - ``crr_removal.<type>.objlim``
      - float
      - The minimum contrast between the cosmic ray and the object.
    * - ``crr_removal.<type>.sigclip``
      - float
      - The number of sigma to use for the sigclip algorithm.
    * - ``crr_removal.<type>.niter``
      - integer
      - The number of iterations to use for the algorithm.

Background Subtraction Configuration
------------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``background_sub.pairs``
      - see :ref:`the tutorial <ab>`
      - The pairs of frames to use for the background subtraction when dithering.

Science Frames Configuration
----------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``science.skip_science``
      - boolean
      - true if the science frames should be skipped (only standard star reduction), false otherwise.
    * - ``science.science_dir``
      - string
      - The directory where the science frames are located (this directory should not contain any other files).
    * - ``science.exptime``
      - float
      - The exposure time of the science frames.
    * - ``science.airmass``
      - float
      - The airmass of the science frames (if several frames, the average airmass).

Standard Star Configuration
---------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``standard.skip_standard``
      - boolean
      - true if the standard star frames should be skipped (only science reduction), false otherwise.
    * - ``standard.standard_dir``
      - string
      - The directory where the standard star frames are located (this directory should not contain any other files).
    * - ``standard.exptime``
      - float
      - The exposure time of the standard star frames.
    * - ``standard.airmass``
      - float
      - The airmass of the standard star frames (if several frames, the average airmass).
    * - ``standard.starname``
      - string
      - The name of the standard star (logging purposes only).
    * - ``standard.flux_file_path``
      - string
      - The path to the file containing the flux of the standard star **in ab magnitudes**.

Arc Lamp Configuration
----------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``arc.arc_dir``
      - string
      - The directory where the arc-lamp frames are located (this directory should not contain any other files).
    * - ``combine_arcs.skip_bias``
      - boolean
      - true if the bias subtraction should be skipped for the arc-lamp frames, false otherwise.

Wavelength Calibration Configuration
------------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``wavecalib.offset_middle_cut``
      - integer
      - Normally the software uses the middle of the arc-lamp frame to find the lines, but if the lines are not in the middle, this parameter can be used to offset the middle.
    * - ``wavecalib.pixel_cut_extension``
      - integer
      - The number of pixels to average over when taking the 1d spectrum of the arc-lamp frame.
    * - ``wavecalib.arcline_start``
      - integer
      - The starting spatial pixel of the lines (useful to avoid noisy edges).
    * - ``wavecalib.arcline_end``
      - integer
      - The ending spatial pixel of the lines (useful to avoid noisy edges).
    * - ``wavecalib.jump_tolerance``
      - float
      - The tolerance for the jump in the lines.
    * - ``wavecalib.center_guess_pixtable``
      - string
      - The path to the file containing the lines and their wavelengths from the :ref:`pylongslit_identify_arcs procedure <identify>`.
    * - ``wavecalib.FWHM``
      - float
      - The FWHM guess of the lines in the arc-lamp frame.
    * - ``wavecalib.TOL_MEAN``
      - float
      - The tolerance for the correction of the line center compared to the ones in the :ref:`pixtable <identify>`.
    * - ``wavecalib.TOL_FWHM``
      - float
      - The tolerance for the correction of the line FWHM compared to the initial guess.
    * - ``wavecalib.REIDENTIFY_R2_TOL``
      - float
      - Threshold for the R² value of the fit for reidentified lines.
    * - ``wavecalib.ORDER_WAVELEN_1D``
      - integer
      - The order of the polynomial used to fit the wavelength solution.
    * - ``wavecalib.ORDER_SPECTRAL_TILT``
      - integer
      - The order of the polynomial used to fit the spectral tilt.
    * - ``wavecalib.ORDER_SPATIAL_TILT``
      - integer
      - The order of the polynomial used to fit the spatial tilt.
    * - ``wavecalib.TILT_TRACE_R2_TOL``
      - float
      - The R² threshold for the fit of the tilt traces.
    * - ``wavecalib.TILT_REJECT_LINE_FRACTION``
      - float
      - The fraction of bad fits at when to abort a line trace.
    * - ``wavecalib.SPATIAL_R2_TOL``
      - float
      - The R² threshold for the fit of the spatial direction.
    * - ``wavecalib.reuse_reided_lines``
      - boolean
      - true if use the reidentified lines that are saved in the output directory, false otherwise.
    * - ``wavecalib.reuse_1d_sol``
      - boolean
      - true if use the 1d solution that is saved in the output directory, false otherwise.
    * - ``wavecalib.reuse_line_traces``
      - boolean
      - true if use the line traces that are saved in the output directory, false otherwise.
    * - ``wavecalib.reuse_2d_tilt_fit``
      - boolean
      - true if use the 2d tilt fit that is saved in the output directory, false otherwise.

Sky Subtraction Configuration
-----------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``sky.sigma_cut``
      - float
      - the number of sigma to use to reject outliers in sky-fitting.
    * - ``sky.sigma_clip_iters``
      - integer
      - the number of iterations to use for the sigma-clip algorithm for sky-fitting.
    * - ``sky.fit_order``
      - integer
      - the order of the polynomial to fit to the sky.

Object Tracing Configuration
----------------------------

**Note: the parameters below live under either** ``trace.object`` **or**
``trace.standard`` **.**

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``trace.<type>.spectral_pixel_extension``
      - integer
      - Number of pixels to average the 1D spectrum over when fitting the trace.
    * - ``trace.<type>.fwhm_guess``
      - float
      - Spatial FWHM initial guess.
    * - ``trace.<type>.fwhm_thresh``
      - float
      - Allowed deviation of the fitted FWHM from the guess.
    * - ``trace.<type>.center_thresh``
      - float
      - Allowed deviation of the fitted center from the manual center.
    * - ``trace.<type>.SNR``
      - float
      - Required signal-to-noise ratio for detecting the trace.
    * - ``trace.<type>.fit_order_trace``
      - integer
      - Polynomial order to fit the center trace.
    * - ``trace.<type>.fit_order_fwhm``
      - integer
      - Polynomial order to fit the FWHM trace.
    * - ``trace.<type>.fit_R2``
      - float
      - Minimum R² for accepting fits.
    * - ``trace.<type>.use_bspline_obj``
      - boolean
      - Use a bspline for the center trace.
    * - ``trace.<type>.use_bspline_fwhm``
      - boolean
      - Use a bspline for the FWHM trace.
    * - ``trace.<type>.knots_bspline``
      - integer
      - Number of bspline knots.
    * - ``trace.<type>.model``
      - string
      - Profile model to fit; allowed values: ``"Gaussian"`` or ``"Cauchy"``.

Object Trace Cloning Configuration
----------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``obj_trace_clone.archived_spec_root``
      - string
      - The path to the 1d spectrum to clone.
    * - ``obj_trace_clone.frame_root``
      - string
      - The path to the 2d frame to clone the 1d spectrum onto.

Sensitivity Function Configuration
----------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``sensfunc.fit_order``
      - integer
      - The order of the polynomial to fit to the sensitivity function.
    * - ``sensfunc.use_bspline``
      - boolean
      - true if a bspline should be used to fit the sensitivity function, false otherwise.
    * - ``sensfunc.knots_bspline``
      - integer
      - The number of knots in the bspline.

Flux Calibration Configuration
------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``flux_calib.path_extinction_curve``
      - string
      - The path to the extinction curve for the observatory **in AB magnitudes**.

Spectral Combination Configuration
----------------------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``combine``
      - see :ref:`the tutorial <combine_spec>`
      - Object names as keys with arrays of filenames as values for frames to combine.

Developer Configuration
-----------------------

.. list-table:: 
    :header-rows: 1
    :widths: 30 20 50

    * - Parameter Path
      - Data Type
      - Description.
    * - ``developer.debug_plots``
      - boolean
      - true if debug plots should be shown, false otherwise (for development only).
    * - ``developer.verbose_print``
      - boolean
      - true if verbose print should be used, false otherwise (for development only).

-----------------------    

:ref:`General Notes on using the pipeline <general_notes>` 

First pipeline step → :ref:`Bias subtraction <bias>`
