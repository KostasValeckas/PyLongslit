.. _reduce:

Reduction of observations
==========================

Quick start
------------

The reduction of the observations is done by calling the command:

.. code-block:: bash

    pylongslit_reduce PATH_TO_CONFIG_FILE

This procedure reads the raw observation files (science and/or standard star frames),
and then applies the following steps:

#. :ref:`Bias subtraction. <bias>`
#. :ref:`Dark current subtraction <dark>` (if dark frames are provided).
#. :ref:`Flat-fielding. <flat>`

The products are written to the ouput directory specified in the configuration file, 
with the names ``"reduced_science_ORIGINAL_NAME"`` or ``"reduced_standard_ORIGINAL_NAME"``,
depending on the frame type. All further operations up to :ref:`1d spectrum extraction <extract_1d>` 
will from now on be performed on these frames, and the raw frames will not be used anymore.
The reduced frames will be altered in place. This also means that **if you in further steps do 
any operations on the reduced frames that you regret, you can always reset by running the reduction procedure again.**

5 fits headers are added to the reduced frames. These are non-standard headers,
and can be used only in the context of the PyLongslit software. The headers are:

``"CRRREMOVD"`` - true if :ref:`cosmic rays have been removed <crr>` from the frame.

``"BCGSUBBED"`` - true if :ref:`A-B sky background subtraction <ab>` has been performed on the frame.

``"CROPY1"`` and ``"CROPY2"`` - the y-pixel range of the frame after :ref:`cropping <crop>`.

``"SKYSUBBED"`` - true if the :ref:`modelled sky background <sky>` has been subtracted from the frame.

Quality Assesment
------------------

The reduced frames will be shown on the screen, and you can press ``h`` to normalize the frames.
If you have detected no issues in the bias, dark and flat-frame procedures, you should not expect to see any
issues in the reduced frames, assuming if your raw object data is healthy. The reduced frames at this point
should resemble the raw frames very much to the naked eye. You should also check that the error
image has reasonable values (in the order of :math:`\sqrt{counts}`). This can be done by 
hoovering the mouse over the image, and the error value will be shown in the corner of the window
(which corner can depends on your system).




Parameter options
------------------

The only aprameters that can alter the reduction routine are

.. code::

    "science" : {
        "skip_science": false, # true if you are only reducing standard star frames
    },

    "standard" : {
        "skip_standard": false, # true if you are only reducing science frames
    },

