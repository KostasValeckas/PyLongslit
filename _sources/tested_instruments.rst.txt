.. _tested_instruments:

Tested Instruments and Configurations
=====================================

As described in the :ref:`pipeline overview <index>`, the pipeline is designed 
to be instrument-independent, as long as some primary assumptions are met 
(see the :ref:`pipeline overview <index>` for details).

However, the :ref:`configuration files <conf>` for an instrument 
setup can be viewed as an instrument implementation, as for a fixed 
instrument setup, most of the parameters in the :ref:`configuration file <conf>`
will be constant and very little will need to be changed between different datasets.
Furthermore, resources like the products of :ref:`initial arc line identification <identify>` and 
:ref:`extinction curves <sensfunction>` can be re-used. We therefore provide an 
overview of already tested instrument setups with their resources and configuration files 
in the hope that these will be useful for users of the software.

.. list-table::
    :header-rows: 1
    :widths: 25 25 25 25 25 25

    * - Instrument
      - Telescope
      - Disperser
      - Pixtable (Initial Arc Lines Guess)
      - Extinction Curve
      - Configuration File Example

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #4
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/alfosc_grating4_hene_pixtable.dat>`__ 
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__ 
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/ALFOSC/grism4.json>`__

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #18
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/alfosc_grating18_thar_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/ALFOSC/grism18.json>`__

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #19
      - `Link (ThAr) <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/alfosc_grating19_thar_pixtable.dat>`__ `Link (HeNe) <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/alfosc_grating19_hene_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/ALFOSC/grism19.json>`__

    * - ALFOSC
      - Nordic Optical Telescope
      - Grism #20
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/alfosc_grating20_thar_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/ALFOSC/grism20.json>`__

    * - OSIRIS
      - Gran Telescopio Canarias
      - R1000B
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/osiris_r1000b_hgar_ne_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/OSIRIS/r1000b.json>`__

    * - OSIRIS
      - Gran Telescopio Canarias
      - R1000R
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/OSIRIS_R1000R_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/lapalma.dat>`__ `Source <https://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/wht_instr/pfip/node244.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/OSIRIS/r1000r.json>`__

    * - FORS2
      - Very Large Telescope
      - 300I
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/pixtables/fors2_test_pixtable.dat>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/database/extinction_curves/paranal.dat>`__ `Source <https://www.aanda.org/articles/aa/full_html/2011/03/aa15537-10/T4.html>`__
      - `Link <https://github.com/KostasValeckas/PyLongslit_dev/blob/main/example_configuration_files/FORS2/300i.json>`__

Contributing new instrument configurations
-------------------------------------------

We encourage users to contribute configurations for new instruments or instrument setups. 
If you have successfully used PyLongslit with an instrument not listed above, please consider 
sharing your configuration files and resources with the community.

To contribute a new instrument configuration, please provide:

1. A working configuration file for your instrument setup.
2. A pixel table (arc line identification file) if available.
3. An appropriate extinction curve for your observatory.
4. A brief description of any instrument-specific considerations or parameter choices.

to kostas.valeckas@nbi.ku.dk .

Your contribution will help expand the software's utility for the broader astronomical community.

