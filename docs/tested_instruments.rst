.. _tested_instruments:

Tested Instruments
==================

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
    :widths: 25 15 25 20 15

    * - Instrument
      - Disperser
      - Pixtable (Initial Lines)
      - Extinction Curve
      - Configuration File
    * - ALFOSC (Nordic Optical Telescope)
      - SX-Grism1
      - pixtable_sx1000.fits
      - extcurve_sx1000.dat
      - `config_alfosc.yaml <configs/config_alfosc.yaml>`__
    * - AstroCam
      - AC-Prism2
      - pixtable_ac200.fits
      - extcurve_ac200.dat
      - `config_astrocam.yaml <configs/config_astrocam.yaml>`__
    * - OptiScope
      - OS-Grating3
      - pixtable_os500.fits
      - extcurve_os500.dat
      - `config_optiscope.yaml <configs/config_optiscope.yaml>`__