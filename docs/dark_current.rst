.. _dark: 

Dark current subtraction
=========================

.. note::

    It is possible that the dark current is negligible for your detector,
    and you can skip this step. Check the documentation of your instrument.

    **Dark current subtraction is not performed in any of the tutorial datasets.**

The dark current estimation is implemented differently than the rest of the 
calibration steps, and **does not need to be called explicitly**.

Instead, if you have dark frames for your flat frames, science object frames
and/or standard star frames, you have to create a directory named "dark"
in the raw data directories and place the dark frames there.

For example, if your raw science frame directory is called:

.. code-block:: bash

    /home/Documents/science

Then you have to create a directory:

.. code-block:: bash

    /home/Documents/science/dark

And place the dark frames there. 

This is only possible for the flat, science and standard star frames, as for the 
arc frames we are only interested in the position of the lines, and 
subtracting the dark current would not change it. Furthermore, most arc frames
are exposed for a very short time, so the dark current is negligible. This is 
also mostly the case for the flat files, but the option is implemented if 
your detector has a significant dark current.

**!IMPORTANT!** The dark frames should have the same exposure time as the
science, flat or standard star frames. This is also the reason why they 
have to be distributed in the same directories as the frames they are
calibrating.

**There are no parameters to set for the dark current estimation.** The 
software takes a bias-subtracted median dark frame from the frames provided, and subtracts it
from the corresponding frames. Technically, bias subtraction could be skipped for dark frames, 
and then the median dark could be treated as a combination of the dark current and the bias level, 
but we choose to subtract the bias separately, as we believe that the bias estimation from bias frames
is more correct, as it is common that more bias frames are present than dark frames. Furthermore, treating 
bias and dark current separately allows us to invoke the overscan regions, allowing to correct for 
the bias drift between the individual frames (see :ref:`bias documentation <bias>`).

For users new to data reduction - short introduction to dark current
---------------------------------------------------------------------

CCD detectors accumulate signal even when no light is present. This signal
is called the dark current, and is caused by thermal excitation of the electrons
in the detector. The dark current is usually very low, but can be significant
for long exposure times, especially for non-cooled CCD detectors. The dark current is estimated by taking a series of dark frames - frames with no incoming light with the same exposure time as the observations. The dark current is then subtracted
from the observations.

-----------------------

:ref:`Bias subtraction <bias>` ← Previous pipeline step  

Next pipeline step → :ref:`Combining arc frames <combine_arcs>`

:ref:`General Notes on using the pipeline <general_notes>` 

:ref:`General info on the configuration file <conf>`