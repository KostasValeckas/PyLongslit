.. PyLongslit documentation master file, created by
   sphinx-quickstart on Fri Jun 21 11:18:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyLongslit's documentation page
======================================

.. image:: https://joss.theoj.org/papers/81b63ae8448434cef0857dd835f1dcc1/status.svg
   :target: https://joss.theoj.org/papers/81b63ae8448434cef0857dd835f1dcc1

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15091603.svg
   :target: https://doi.org/10.5281/zenodo.15091603

.. image:: https://img.shields.io/badge/arXiv-astro--ph-blue
   :target: https://arxiv.org/abs/2504.00973


PyLongslit is a simple manual Python pipeline for processing of astronomical long-slit spectra 
recorded with CCD detectors. At the current version, the software can produce following products:

#. A calibrated 2D spectrum in counts/wavelength.
#. A 1D spectrum extracted from the 2D spectrum in counts/wavelength (for point-like objects).
#. A flux-calibrated 1D spectrum in :math:`\text{erg}/\text{s}/\text{cm}^2/\text{Å}` (for point-like objects).

The software is designed to be instrument-independent, and
is **designed to work with data that has the following characteristics**:

#. The data is taken by a long-slit spectrograph. I.e. the data is 2D, with spatial
   information along one axis and spectral information along the other.
#. The data is recorded with a single CCD detector (or a mosaic of CCD detectors that are treated as one).
#. The data is in FITS format.
#. The data is in counts, and has not been reduced in any way.
#. Bias, flat-field and arc-lamp calibration frames are available. For the arc-lamp, accompanying
   data is available with a list of lines and their wavelengths, with the lines identified in the arc-lamp spectrum.
   *(Wavelength-calibrating using sky-lines should be fine aswell, but it is not yet a tested approach).*

If these conditions are met, the software should be able to reduce the data to 2D and 1D spectra
in counts/wavelength. For further processing, you will also need to acquire a standard star observation
with a reference spectrum, and an extinction curve for your observatory.

If your data does not meet these conditions, you can still try to use the software, but
please know that the software is not designed/tested for data does not meet the above conditions.

**Principles behind the software:**

#. We strive for **simplicity**. *The main motivation for the software is to create a tool that is
   easy to understand and use, specially for novice astronomers and students. 
   There are many high-fidelity software solutions that are more sophisticated and 
   precise than PyLongslit, and we do not aim to out-perform them. Our goal is to
   create an entry-level pipeline that is approachable.*
#. We prioritize **manual execution** over automation. *We believe that a manual pipeline
   is easier to understand and troubleshoot than an automated one. We also believe that
   manual execution is
   more educational, as it forces the user to understand the steps that are being taken.
   We accept the trade-off of this pipeline being only useful for single-object-at-a-time
   processing, and do not compromise on this principle.*
#. We strive for **transparency**. *We look at documentation as an active part of the code.
   No line of code is deemed complete if it is not documented appropriately.
   We specially aim to disclose the limitations of the software, and to provide
   guidance on how to interpret the results - specially through carefully designed quality-assesment plots.*
#. We strive for **robustness**. *We pay special attention to error handling and to
   providing guidance on how to troubleshoot common issues. We commit to providing 
   support to users who are struggling with the software to our best ability.*


**Citing the software**

If you use PyLongslit in your research, please cite the software using the following:

*Valeckas, K., Fynbo, J., Krogager, J.-K., & Elm Heintz, K. (2025). PyLongslit. Zenodo. https://doi.org/10.5281/zenodo.15091602*

Bibtex:


.. code:: 

   @misc{valeckas2025pylongslit,
      author    = {Valeckas, K. and Fynbo, J. and Krogager, J.-K. and Elm Heintz, K.},
      title     = {PyLongslit},
      year      = {2025},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.15091602},
      url       = {https://doi.org/10.5281/zenodo.15091602}
   }

-----------------------------------------------

In this documentation, we provide guidance on installation and usage of the software.


.. toctree:: 
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   develop
   uncertainties
   contact

