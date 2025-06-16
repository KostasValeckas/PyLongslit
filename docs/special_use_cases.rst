Special Use Cases
=================

In this section, we provide special use-case examples to document and explore 
the versatility of the PyLongslit software. All users are welcome to contribute
to this section by providing their own use-case examples, which will be added to the documentation
(see :ref:`developing guidelines <developer>` for more information on how to contribute).

Extracting closely spaced spectra
---------------------------------

With a longslit instrument, it is possible to obtain distinct spectra of closely 
spaced sources within the same object, such as lenssed quasars. The spectra in such cases are placed so 
closely that automatic modelling/fitting routines will sometimes fail. The software can deal with such cases, but a fair amount of manual work is required.
The folloing describes how to extract distinct 1d spectra of very closely spaced sources. 

.. note::
    Before you execute the following steps, make sure you have tried to use the software
    on a more simple case, such as the :ref:`tutorial data <tutorial_data>`. You will 
    need a basic understanding of the routines in order to perform the following steps.


1. **Follow the regular** :ref:`pipeline steps <tutorial>` **up to** :ref:`object tracing <objtrace>`. 
2. **Create copies of the reduced files such that for every exposure, you have a copy corresponding to every spectra you want to extract.**
    For example, if you have 3 reduced exposures of the same object:

    .. code-block:: bash

        ├── output_directory
        │   ├── reduced_science_exposure_1.fits
        │   ├── reduced_science_exposure_2.fits
        │   ├── reduced_science_exposure_3.fits

    And within your object, you have two closely spaced spectra A and B:

    .. image:: pictures/close_trace.png
        :width: 100%
        :align: center

    Then the copied filenames could look like this:

    .. code-block:: bash

        ├── output_directory
        │   ├── reduced_science_exposure_1_A.fits
        │   ├── reduced_science_exposure_1_B.fits
        │   ├── reduced_science_exposure_2_A.fits
        │   ├── reduced_science_exposure_2_B.fits
        │   ├── reduced_science_exposure_3_A.fits
        │   ├── reduced_science_exposure_3_B.fits

    **Notice** that the files witout the ``_A`` or ``_B`` suffix have been deleted. 
    In the later :ref:`object tracing <objtrace>`, the software will connect the object traces
    to the reduced files by the filename, so it is important that the filenames
    are unique for every traced spectra.

3. **Run the** :ref:`manual object tracing <man_trace>` **routine on all of the copied files**. 
    (Run the regular :ref:`object tracing <objtrace>` for the standard star if you are using one - skip 
    the science files in the regular :ref:`object tracing <objtrace>` by pressing ``q`` at every science spectra.)

4. **Extract the distinct spectra by running the (important)** :ref:`simple extraction <sum>` **routine on the copied files**.
    The reason for running the simple extraction is that the simple extraction puts hard box-like boundaries on the extraction 
    region, which is needed for the closely spaced spectra. The hard boundaries prevent the spectra from ''bleeding'' into another. You might need to be conservative with what FWHM guess you use. 
    The QA plot from the :ref:`simple extraction <sum>` can be used to asses if the extraction region is correct.

5. **Flux the spectra by running the** :ref:`flux calibration <flux_calibrate>` **routine on the copied files**.

6. **Lastly** :ref:`combine the fluxed spectra <combine_spec>` **according to the distinct sources**.

    For the example above with distinct sources A and B, the relevant part of the :ref:`configuration file <conf>` would look like this:
    .. code:: 

        "combine": {
            "object_A": ["reduced_science_exposure_1_A.fits", "reduced_science_exposure_2_A.fits", "reduced_science_exposure_3_A.fits"],
            "object_B": ["reduced_science_exposure_1_B.fits", "reduced_science_exposure_2_B.fits", "reduced_science_exposure_3_B.fits"]
        }