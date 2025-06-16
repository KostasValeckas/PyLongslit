.. _tested_instruments:

Tested Instruments and Configurations
=====================================

As described in the :ref:`pipeline overview <index>`, the pipeline is designed 
to be instrument-independent, as long as some primary assumptions are met 
(see the :ref:`pipeline overview <index>` for details).

However, the :ref:`configuration files <configuration_file>` for an instrument 
setup can be viewed as an instrument-implementation, as for a fixed 
instrument setup, most of the parameters in the :ref:`configuration file <configuration_file>`
will be constant and very little will need to be changed between different datasets.
Furthermore, resources like the products of :ref:`initial arc line identification <identify>` and 
:ref:`extinction curves <sensfunction>` can be re-used. We therefore provide an 
overview of already tested instrument setups with their resources and configuration files 
in hope that these will be useful for users of the software.

.. list-table::
    :header-rows: 1
    :widths: 25 25 25 25 25 25

    * - Instrument
      - Telescope
      - Disperser
      - Pixtable (Initial Lines)
      - Extinction Curve
      - Configuration File

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #4
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #18
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #19
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #20
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - OSIRIS
      - Gran Telescopio Canarias
      - R1000B
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - OSIRIS
      - Gran Telescopio Canarias
      - R1000R
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - OSIRIS
      - Gran Telescopio Canarias
      - R2500I
      - Link TBD
      - LINK TBD
      - LINK TBD

    * - FORS2
      - Very Large Telescope
      - 300I
      - Link TBD
      - LINK TBD
      - LINK TBD

