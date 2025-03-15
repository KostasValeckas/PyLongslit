Configuration file
==================

The configuration file is a ``.json`` file that contains all the parameters
that are needed to run the software. The file is divided into sections, 
each corresponding to a different step in the pipeline.

Every pipeline-command takes the confguration file as input. From the 
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
for future runs. Furthermore, most of parameters should be somewhat constant 
between different instruments. This means that for a new instrument, you can
try to **reuse the configuration file from the tutorials**, and change out
only the parameters that are corrupting the run.

The software has a method for checking the configuration file for errors 
(it checks if the provided file paths exist, the raw data directories have files in them, some sanity-checks 
on the parameters, etc.). The method can be called by:

.. code:: bash

    pylongslit_check_config PATH_TO_CONFIG_FILE

It is not mandatory to use this method, but highly recommended before 
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
            "skip_spacial": false,
            "knots_spectral_bspline": 70,
            "degree_spectral_bspline": 3,
            "knots_spacial_bspline": 4,
            "degree_spacial_bspline": 3,
            "R2_spacial_bspline": 0.4

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
            "SPACIAL_R2_TOL": 0.97,
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

A brief explanation of every parameter (see the :ref:`tutorial <tutorial>` for more detailed explanation 
for every step):

.. code:: 

    {
        "instrument": {
            "name": # The name of the instrument, simply for logging purposes
            "disperser": # The disperser used, simply for logging purposes
        },

        "detector": {
            # the below sizes are used to check that all raw data has the same size
            "xsize": # The number of pixels along the x axis
            "ysize": # The number of pixels along the y axis
            "dispersion": {
                "spectral_dir": # The direction of the spectral axis in raw data, either "x" or "y"
                "wavelength_grows_with_pixel": # true if the wavelength increases with pixel number for the spectral direction
                # given in the parameter above, false otherwise
            },
            "gain": # detector gain in electrons per count (ADU)
            "read_out_noise": # read-out noise in electrons
            "overscan" : {
                "use_overscan": # true if overscan is to be used, false otherwise (then the bias is estimated only from bias frames)
                "overscan_x_start": # The starting pixel of the overscan region along the x axis
                "overscan_x_end": # The ending pixel of the overscan region along the x axis
                "overscan_y_start": # The starting pixel of the overscan region along the y axis
                "overscan_y_end": # The ending pixel of the overscan region along the y axis
            }
        },

        "data": {
            "raw_data_hdu_index": # Index of the HDU in the raw data FITS file that contains the data
            # (usually 0 for single-extension FITS files, 1 for multi-extension FITS files)
        },

        "bias": {
            "bias_dir": # The directory where the bias frames are located (this directory may not countain any other files)
            "bootstrap_errors": # true if bootstrapping should be used to estimate the error in the bias frames, false otherwise
            # (bootstrapping takes longer, but is more accurate)
        },

        "flat": {
            "flat_dir": # The directory where the flat-field frames are located (this directory may not countain any other files)
            "bootstrap_errors": # true if bootstrapping should be used to estimate the error in the flat-field frames, false otherwise
            # (bootstrapping takes longer, but is more accurate)
            "skip_spacial": # true if slit-illumination correction should be skipped, false otherwise (see the tutorial for more info)
            "knots_spectral_bspline": # The number of knots in the bspline when fitting the detector spectral response
            "degree_spectral_bspline": # The degree of the bspline when fitting the detector spectral response
            "knots_spacial_bspline": # The number of knots in the bspline when fitting the detector spacial response
            "degree_spacial_bspline": # The degree of the bspline when fitting the detector spacial response
            "R2_spacial_bspline": # The rejection threshold for the bspline when fitting the detector spacial response

        },

        "output": {
            "out_dir": # The directory where the output files should be saved (should not contain any other files)
        },

        # the below 2 sets of parameters have same meaning, but are used for science and standard frames, respectively (see tutorial)
        "crr_removal" : {
            "science":{
                "frac": # The fraction of sigclip to use for the lower limit of the contrast detection (see tutorial).
                "objlim": # The minimum contrast between the cosmic ray and the object (see tutorial).
                "sigclip": # The number of sigma to use for the sigclip algorithm (see tutorial).
                "niter": # The number of iterations to use for the algorithm (see tutorial).
            },
            "standard":{
                "frac": # The fraction of sigclip to use for the lower limit of the contrast detection (see tutorial).
                "objlim": # The minimum contrast between the cosmic ray and the object (see tutorial).
                "sigclip": # The number of sigma to use for the sigclip algorithm (see tutorial).
                "niter": # The number of iterations to use for the algorithm (see tutorial).
            }
        },

        "background_sub" : {
            "pairs": {
                # The pairs of frames to use for the background subtraction (B is subtracted from A)
                "1": { # The pair number (start with 1 and increment by 1 for every pair)
                    "A": # filename (just the name, not the full path, ex. "filename.fits")
                    "B": # filename (just the name, not the full path, ex. "filename.fits")
                },
                "2": { # The pair number
                    "A": # filename (just the name, not the full path, ex. "filename.fits"),
                    "B": # filename (just the name, not the full path, ex. "filename.fits")
                }
            }
        },

        "science" : {
            "skip_science": # true if the science frames should be skipped (only standard star reduction), false otherwise
            "science_dir": # The directory where the science frames are located (this directory may not countain any other files)
            "exptime": # The exposure time of the science frames
            "airmass": # The airmass of the science frames (if several frames, the average airmass)
        },

        "standard" : {
            "skip_standard": # true if the standard star frames should be skipped (only science reduction), false otherwise
            "standard_dir": # The directory where the standard star frames are located (this directory may not countain any other files)
            "exptime": # The exposure time of the standard star frames
            "airmass": # The airmass of the standard star frames (if several frames, the average airmass)
            "starname": # The name of the standard star (logging purposes only)
            "flux_file_path": # The path to the file containing the flux of the standard star IN AB MAGNITUDES (see tutorial)

        },

        "arc" : {
            "arc_dir": # The directory where the arc-lamp frames are located (this directory may not countain any other files)
        },

        "combine_arcs" : {
            "skip_bias": # true if the bias subtraction should be skipped for the arc-lamp frames, false otherwise
        },

        # the wavelength procedure is the most complex, and has the most parameters. 
        # The descritions here won't make much sense without the tutorial, so please see the tutorial if you are new to the software.
        "wavecalib" : {
            "offset_middle_cut": # Normally the software uses the middle of the arc-lamp frame to find the lines, but if the lines are not in the middle,
            # this parameter can be used to offset the middle
            "pixel_cut_extension": # The number of pixels to average over when taking the 1d spectrum of the arc-lamp frame
            "arcline_start": # The starting spatial pixel of the lines (useful to avoid noisy edges)
            "arcline_end": # The ending spatial pixel of the lines (useful to avoid noisy edges)
            "jump_tolerance": # The tolerance for the jump in the lines
            "center_guess_pixtable": # The path to the file containing the lines and their wavelengths from the  pylongslit_identify_arcs procedure
            "FWHM": # The FWHM guess of the lines in the arc-lamp frame
            "TOL_MEAN": # The tolerance for the correction of the line center compared to the ones in the pixtable
            "TOL_FWHM": # The tolerance for the correction of the line FWHM compared to the initial guess
            "REIDENTIFY_R2_TOL": # Threshold for the R2 value of the fit for reidentified lines
            "ORDER_WAVELEN_1D": # The order of the polynomial used to fit the wavelength solution
            "ORDER_SPECTRAL_TILT": # The order of the polynomial used to fit the spectral tilt
            "ORDER_SPATIAL_TILT": # The order of the polynomial used to fit the spatial tilt
            "TILT_TRACE_R2_TOL": # The R2 threshold for the fit of the tilt traces
            "TILT_REJECT_LINE_FRACTION": # The fraction of bad fits at when to abort a line trace
            "SPACIAL_R2_TOL": # The R2 threshold for the fit of the spatial direction
            "reuse_reided_lines": # true if use the reidentified lines that are saved in the output directory, false otherwise
            "reuse_1d_sol": # true if use the 1d solution that is saved in the output directory, false otherwise
            "reuse_line_traces": # true if use the line traces that are saved in the output directory, false otherwise
            "reuse_2d_tilt_fit": # true if use the 2d tilt fit that is saved in the output directory, false otherwise
        },

        "sky" : {
            "sigma_cut": # the number of sigma to use to reject outliers in sky-fitting
            "sigma_clip_iters": # the number of iterations to use for the sigma-clip algorithm for sky-fitting
            "fit_order": # the order of the polynomial to fit to the sky
        },

        # the two below sets of parameters have the same meaning, but are used for science and standard star frames, respectively
        "trace" : {
            "object": {
                "spectral_pixel_extension": # The number of pixels to average the 1d spectrum over when fitting the object trace (see tutorial)
                "fwhm_guess": # The spatial FWHM guess of the object 
                "fwhm_thresh": # The threshold by which the fitted FWHM can deviate from the guess
                "center_thresh": # The threshold by which the fitted center can deviate from the manually set center
                "fit_order_trace": # The order of the polynomial to fit to the object center trace
                "fit_order_fwhm": # The order of the polynomial to fit to the object FWHM trace
                "fit_R2": # The R2 threshold for the fit of the object trace
                "use_bspline_obj": # true if a bspline should be used to fit the object center trace, false otherwise (should only be used if regular fit fails)
                "use_bspline_fwhm": # true if a bspline should be used to fit the object FWHM trace, false otherwise (should only be used if regular fit fails)
                "knots_bspline": # The number of knots in the bspline
                "model": # The model to use for the object trace ("Gaussian" or "Cauchy") (see tutorial)
            },
            "standard": {
                "spectral_pixel_extension": # The number of pixels to average the 1d spectrum over when fitting the object trace (see tutorial)
                "fwhm_guess": # The spatial FWHM guess of the object 
                "fwhm_thresh": # The threshold by which the fitted FWHM can deviate from the guess
                "center_thresh": # The threshold by which the fitted center can deviate from the manually set center
                "fit_order_trace": # The order of the polynomial to fit to the object center trace
                "fit_order_fwhm": # The order of the polynomial to fit to the object FWHM trace
                "fit_R2": # The R2 threshold for the fit of the object trace
                "use_bspline_obj": # true if a bspline should be used to fit the object center trace, false otherwise (should only be used if regular fit fails)
                "use_bspline_fwhm": # true if a bspline should be used to fit the object FWHM trace, false otherwise (should only be used if regular fit fails)
                "knots_bspline": # The number of knots in the bspline
                "model": # The model to use for the object trace ("Gaussian" or "Cauchy") (see tutorial)
            }
        },

        # object trace cloning is used to clone the object trace from one frame to another
        "obj_trace_clone" : {
            "archived_spec_root": # The path to the 1d spectrum to clone
            "frame_root": # The path to the 2d frame to clone the 1d spectrum onto
        },

        "sensfunc": {
            "fit_order": # The order of the polynomial to fit to the sensitivity function
            "use_bspline": # true if a bspline should be used to fit the sensitivity function, false otherwise (should only be used if regular fit fails)
            "knots_bspline": # The number of knots in the bspline
        },

        "flux_calib": {
            "path_extinction_curve": # The path to the extinction curve for the observatory ! IN AB MAGNITUDES !
        },

        "combine": {
            #"name" : ["filename1.fits", "filename2.fits", ... ] # The name of the object and the frames to combine for the object
            # several objects can be added 
        },

        # the below parameters are for developer purposes only, activates aggresive printing and plotting
        # code may crash if these are set to true - only for debugging purposes for developers, but might be 
        # useful when adapting the configuration file for a new instrument
        "developer": {
            "debug_plots": # true if debug plots should be shown, false otherwise
            "verbose_print": # true if verbose print should be used, false otherwise

        }
    }