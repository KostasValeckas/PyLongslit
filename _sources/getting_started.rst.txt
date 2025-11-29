Getting started
~~~~~~~~~~~~~~~

Execution of the software is highly manual, and therefore the first time 
using it can be a bit overwhelming. The software is designed to be
simple, but it is also designed to be instrument-independent. This means that there are
many parameters that can be set to match your data and setup. 

We provide 2 tutorial datasets that have all the parameters pre-set, so that
you can run the software without having to change anything to get a feel
for how it works.

.. _tutorial_data:

Downloading the tutorial data
----------------------------------------------------

There are two tutorial datasets available. **Execution of either one is enough 
to get familiar with the software.** The two datasets are:

`SDSS_J213510+2728 (from ALFOSC - Nordic Optical Telescope) (ZIP) <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/SDSS_J213510%2B2728>`_


`GQ1218+0832 (from OSIRIS - Gran Telescopio Canarias) (ZIP) <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/GQ1218%2B0832>`_

To complete the tutorial, you will also need to use some complementary files that 
can be `downloaded here (ZIP) <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/database>`_.


**Using git (if you are not familiar with git, just download the ZIP from the link above):** 

You can download the entire tutorial and development repository using git.

SSH (recommended if you plan on developing)...

.. code-block:: bash

    git clone git@github.com:KostasValeckas/PyLongslit_dev.git

... or HTTPS (works too, but you will need to enter your username and password on every pull/push):

.. code-block:: bash

    git clone https://github.com/KostasValeckas/PyLongslit_dev.git


Changing the file paths in the configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As stated above, the configuration files for the tutorials have the parameters pre-set.
**However**, you will need to change all the file paths
in the configuration file to match the location of the tutorial data on your
computer. Everything in the paths from ``SDSS_J213510+2728/..`` or ``GQ1218+0832/..`` will be correct,
but you will need to change the path up to that point to match
the location of the folder on your computer, i.e. the part marked
with bold in the example below:

   **/home/kostas/Documents/PyLongslit_dev/** SDSS_J213510+2728/arcs

For some systems, you will need 
to change the forward slash ``/`` to a backslash ``\`` in the paths.

The above described changes can be made easily by using the *find and replace* functionality in a text editor.

The configuration files in the tutorial and developer repository are placed as shown below:

.. code-block:: bash
   :emphasize-lines: 3,5
   
   PyLongslit_dev
   ├── GQ1218+0832
   │   ├── GQ1218+0832.json
   ├── SDSS_J213510+2728
   │   ├── SDSS_J213510+2728.json


.. _tutorial:

Tutorial
----------------------------------------------------

You can now follow the steps described below to run the software on the tutorial data.

You have to execute the steps exactly in the order they are presented in the
contents table below.


.. toctree:: 
   :maxdepth: 1
   :caption: Contents:

   general_notes
   configuration_file
   bias
   dark_current
   combine_arcs
   identify
   wavecalib
   flat
   reduce
   crr
   AB
   crop
   sky
   show_2dspec
   objtrace
   extract_1d
   sensfunction
   calibrate
   combine_spec