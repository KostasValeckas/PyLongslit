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
to get familiar with the software.**

The tutorial data can be downloaded in two ways: either by cloning the `PyLongslit development
repository  <https://github.com/KostasValeckas/PyLongslit_dev>`_  or by downloading the ZIP files directly from the links below.
We recommend cloning the whole repository (or at least downloading the whole repository as a ZIP) as it will also provide you with additional files needed to complete the tutorial, and in the directory structure that resembles the provided tutorial :ref:`configuration files <conf>`. 
If you prefer just downloading the tutorial data, the links for that are also provided below.

**Cloning using git (if you are not familiar with git, you can download the ZIP from the link below):** 

SSH (recommended - help on SSH keys can be found `here <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_):

.. code-block:: bash

    git clone git@github.com:KostasValeckas/PyLongslit_dev.git

... or HTTPS (works too, but you will need to enter your username and password when dowloading or getting the latest updates to the repository):

.. code-block:: bash

    git clone https://github.com/KostasValeckas/PyLongslit_dev.git

**Downloading a snapshot of the repository as a ZIP file:**

Follow the download link `here <https://github.com/KostasValeckas/PyLongslit_dev/archive/refs/heads/main.zip>`_ .

**Downloading only the tutorial data as ZIP files** (this will not include the complementary files needed to complete the tutorial):

`SDSS_J213510+2728 (from ALFOSC - Nordic Optical Telescope) (ZIP) <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/SDSS_J213510%2B2728>`_


`GQ1218+0832 (from OSIRIS - Gran Telescopio Canarias) (ZIP) <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/GQ1218%2B0832>`_

The missing additional files can be downloaded `here <https://downgit.github.io/#/home?url=https://github.com/KostasValeckas/PyLongslit_dev/tree/main/database>`_.





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