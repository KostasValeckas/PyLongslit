Getting started
~~~~~~~~~~~~~~~

Execution of the software is highly manual, and therefore the first time 
using it can be a bit overwhelming. The software is designed to be
simple, but it is also designed to be instrument-independent. This means that there are
many parameters that can be set to match your data and setup. 

We provide 2 tutorial datasets that have all the prameters pre-set, so that
you can run the software without having to change anything to get a feel
for how it works.

.. _tutorial_data:

Downloading the tutorial data
----------------------------------------------------


`Download As a File (ZIP) <https://github.com/KostasValeckas/PyLongslit_dev/archive/refs/heads/main.zip>`_

**Using git: (if you don't know what git is just download the ZIP from link above)** 

SSH (recommended if you plan on developing)...

.. code-block:: bash

    git clone git@github.com:KostasValeckas/PyLongslit_dev.git

... or HTTPS (works too, but you will need to enter your username and password on every pull/push):

.. code-block:: bash

    git clone https://github.com/KostasValeckas/PyLongslit_dev.git


Changing the file pathes in the configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As stated above, the configuration files for the tutorials have the parameters pre-set.
**However**, you will need to change all the file pathes
in the configuration file to match the location of the tutorial data on your
computer. Everything in the pathes from ``PyLonsglit_dev/..`` will be correct,
but you will need to change the path to the ``PyLongslit_dev`` folder to match
the location of the folder on your computer, i.e. the part marked
with bold in the example below:

   **/home/kostas/Documents/** PyLongslit_dev/SDSS_J213510+2728/arcs

For some systems, you will need 
to change the forward slash ``/`` to a backslash ``\`` in the pathes.

The above described changes can be made easily by using *find and replace* functionality in a text editor.

The configuration files are placed as shown below:

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

You have to excecute the steps exactly in the order they are presented in the
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