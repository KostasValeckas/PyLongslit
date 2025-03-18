Setting up the directory and file structure
===========================================

We have prepped the `tutorial` directory with the necessary files and scripts
to guide you through the reduction of a long-slit spectrum. Everything you will
need is in the `tutorial` directory, and you should not need to move any files
at any time.

If/when reducing own observations, we advise you to copy the `tutorial` directory
and  then replacing/editing relevant files for individual observations. 


The file structure will be broken down in detail in the descriptions 
of the individual steps, but the overall structure is as follows:

.. code-block:: bash

   ├── calibrate.py
   ├── database
   │   ├── lapalma.dat
   │   ├── map-g04-he-1.pdf
   │   ├── map-g04-he-2.pdf
   │   ├── map-g04-ne-1.pdf
   │   ├── map-g04-ne-2.pdf
   │   └── mylines_vac.dat
   ├── extract_1d.py
   ├── extract_science_1d.py
   ├── extract_std_1d.py
   ├── identify.py
   ├── mfeige110.dat
   ├── raw
   │   ├── ALDh120176.fits
   │   ├── ALDh120177.fits
   │   ├── ALDh120178.fits
   │   ├── ALDh120179.fits
   │   ├── ALDh120217.fits
   │   ├── ALDh130384.fits
   │   ├── ALDh130385.fits
   │   ├── ALDh130386.fits
   │   ├── ALDh130387.fits
   │   ├── ALDh130388.fits
   │   ├── ALDh140211.fits
   │   ├── ALDh140219.fits
   │   ├── ALDh140230.fits
   │   ├── ALDh140238.fits
   │   ├── ALDh140351.fits
   │   ├── ALDh140352.fits
   │   ├── ALDh140353.fits
   │   ├── ALDh140354.fits
   │   ├── ALDh140355.fits
   │   ├── ALDh140356.fits
   │   ├── ALDh140357.fits
   │   ├── ALDh140358.fits
   │   ├── ALDh140359.fits
   │   ├── ALDh140360.fits
   │   └── ALDh140361.fits
   ├── rawbias
   │   ├── mkspecbias.py
   │   └── specbias.list
   ├── rawflats
   │   ├── mkspecflat.py
   │   └── specflat.list
   ├── rawscience
   │   ├── crremoval.py
   │   ├── raw_arcs.list
   │   ├── raw_science.list
   │   └── reducescience.py
   ├── rawstd
   │   ├── crremoval.py
   │   ├── raw_arcs.list
   │   ├── raw_std.list
   │   └── reducestd.py
   ├── reduceobs.py
   ├── sensfunction.py
   ├── setup.py
   └── standard.py
