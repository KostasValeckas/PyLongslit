---
title: 'PyLongslit: a simple manual Python pipeline for processing of astronomical long-slit spectra recorded with CCD detectors'
tags:
  - Python
  - astronomy
  - spectroscopy
  - pipelines
authors:
  - name: Kostas Valeckas
    orcid: 0009-0007-7275-0619
    affiliation: 1
  - name: Johan Peter Uldall Fynbo
    orcid: 0000-0002-8149-8298
    affiliation: 2
  - name: Jens-Kristian Krogager
    orcid: 0000-0002-4912-9388
    affiliation: 3
  - name: Kasper Elm Heintz
    orcid: 0000-0002-9389-7413
    affiliation: 2



affiliations:
 - name: Niels Bohr Institute, Copenhagen University, Denmark
   index: 1
  
# - name: Nordic Optical Telescope
#   index: 2

 - name: Cosmic Dawn Center, Niels Bohr Institute, Copenhagen University, Denmark
   index: 2

 - name: Centre de Recherche Astrophysique de Lyon, France
   index: 3

date: 27 March 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

We present a new Python pipeline for processing data from astronomical 
long-slit spectroscopy observations recorded with CCD detectors.

The pipeline is designed to aim for **simplicity**, **manual execution**, **transparency** and **robustness**. The goal for the pipeline is to provide a manual and simple counterpart to the 
well-established semi-automated and automated pipelines. The intended use-cases are **teaching** and **cases where 
automated pipelines fail**. For further elaboration,
please see the [Statement of need](#statement-of-need). 

From raw data, the
pipeline can produce the following output:

- A calibrated 2D spectrum in counts and wavelength for every detector pixel.
- A 1D spectrum extracted from the 2D spectrum in counts per wavelength (for point-like objects).
- A flux-calibrated 1D spectrum in $\frac{\text{erg}}{\text{s} \cdot \text{cm}^2 \cdot \text{Å}}$ (for point-like objects).


The products are obtained by performing standard procedures for
detector calibrations [@handbook ; @Howell_2006], cosmic-ray subtraction [@cr_1 ; @cr_2]
, and 1D spectrum extraction [@Horne_1986 ; @photutils].   

# Statement of need

A natural approach when developing data-processing pipelines is to seek for precision and automation. The trade-off for this 
is code complexity and "black-box" solutions, where the process of the pipeline is often masked, and 
the quality-assessment output is made under the assumption that the user knows how to interpret it. 
In research, this is a reasonable trade-off, as a certain level of user-skill and experience can be assumed. However, 
in a teaching paradigm, simplicity and transparency are often more favorable, even when this means loss of 
precision and automation. The PyLongslit pipeline is designed to rely on simple code and manual execution,
supported by a large array of quality-assessment plots and extensive documentation. The algorithms are designed to produce research-quality results, yet while prioritizing simplicity over high precision. The reason for this is 
to create a robust and transparent pipeline, where every step of the execution is visualized and explained. We see this as being specially valuable in teaching scenarios and for users that 
are new to spectroscopic data processing. Furthermore, we hope that 
the simple coding style will invite users of all skill-levels to contribute to the code.

An early beta-version of the software was user-tested during the [Nordic Optical Telescope](https://www.not.iac.es/) [IDA summer-course 
2024](https://phys.au.dk/ida/events/not-summer-school-2024), where all student groups were able to follow the documentation and successfully process data 
without any significant assistance. 

During the development of software it became apparent that the manual nature of the pipeline is 
also useful for observations where automated pipelines might fail. The PyLongslit pipeline can revert to manual methods instead of using mathematical modelling when estimating the observed object trace on the 
detector. This is especially useful for objects
that have low signal-to-noise ratio, or where several objects are very close to each other on the detector. Furthermore, extraction can be performed with either optimal extraction methods [@Horne_1986], 
or by summing detector counts for a box-like object shape [@photutils] (this can be useful 
for emission-line dominated objects).    


# Pipeline

The figures below describe the pipeline structure. The inspiration for the pipeline 
architecture is taken from the very popular (but no longer maintained) IRAF [@IRAF]. In a broad sense, there
are three stages of the data processing, all explained in separate figures. The diamond shapes in the figures represent different pipeline routines that 
are called directly from the command line, solid arrows are hard-dependencies (must-have), dashed arrows are soft-dependencies (can use) and the rectangles represent input files 
and pipeline products.

![Step 1 - processing raw data. In this step, all the raw observation and calibration frames are used to construct calibrated 2D spectra. After this step, all procedures are performed directly on the calibrated 2D spectra, and the raw frames are no longer used.\label{fig:raw_processing}](raw_processing.png)



![Step 2 - further processing of the calibrated 2D spectra. In this step, the user can deploy cosmic-ray removal, sky-background subtraction, and crop the spectra. All procedures alter the 2D spectra in place. All of the steps are optional, but there are some dependencies — these are described in the figure.\label{fig:further_processing}](further_processing.png)


![Step 3 - 1d spectrum extraction. In this step, objects are traced, extracted, flux calibrated and combined (if several spectra of the same object exist).\label{fig:1d_extraction}](1d_extraction.png)

The pipeline is controlled by a configuration file that has to be passed as an 
argument to every pipeline procedure. The different parameters of the configuration 
file are described in the [documentation](https://kostasvaleckas.github.io/PyLongslit/index.html).


# Evaluation

To test the pipeline for correctness, we run the pipeline on data from two long-slit instruments: [NOT ALFOSC](https://www.not.iac.es/instruments/alfosc/) and [GTC OSIRIS](https://www.gtc.iac.es/instruments/osiris/), and compare the results with the results from the well-established, 
semi-automated PypeIt Python pipeline  [@pypeit:joss_pub ; @pypeit:joss_arXiv ; @pypeit:zenodo]:

![GTC OSIRIS observation of GQ1218+0823.\label{fig:gtc}](gtc_comp.png)

![NOT ALFOSC observation of SDSS_J213510+2728.\label{fig:alfosc}](alfosc_comp.png)

We see good agreement between the two pipelines on both axes, 
for both the extracted spectrum and the noise estimation. For NOT ALFOSC data, 
we see some deviation in the error magnitude and error related to strong sky-lines.
This is due to skipping modelled sky subtraction in the PyLongslit run,  as the
A-B sky background subtraction was sufficient by itself. We calculate the noise
numerically for a cut of the spectrum where the flux is somewhat constant to confirm that
the PyLongslit noise indeed is smaller in magnitude:

![Numerical noise comparison for NOT ALFOSC observation of SDSS_J213510+2728.\label{fig:alfosc_zoom}](not_zoom.png)

We disclose the data and parameters used for both pipeline executions. 

PyLongslit: https://github.com/KostasValeckas/PyLongslit_dev.

PypeIt: https://sid.erda.dk/share_redirect/efiQfMWqqe

# Limitations

As mentioned in the [Statement of need](#statement-of-need), PyLongslit favors
simplicity over high precision. Furthermore, the pipeline is designed to be 
**instrument independent**. Due to these design choices, the pipeline does not account for any instrument-specific phenomena, such as detector fringing and alike. The pipeline will likely be less precise than an instrument-specific pipeline (depending on the implementation of the latter). The code 
is written with focus on **loose-coupling**, and therefore the pipeline code can be used 
as a starting-point for an instrument-specific pipeline.


# Acknowledgements

We thank the participants of the Nordic Optical Telescope IDA summer-course 
2024 for very useful feedback on the software.


# References
