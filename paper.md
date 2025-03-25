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
    affiliation: "1, 2"
  - name: Johan Peter Uldall Fynbo
    orcid: 0000-0002-8149-8298
    affiliation: 3

affiliations:
 - name: Niels Bohr Institute, Copenhagen University
   index: 1
  
 - name: Nordic Optical Telescope
   index: 2

 - name: Cosmic Dawn Center, Niels Bohr Institute, Copenhagen University
   index: 3

date: 24 March 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

We present a new manual Python pipeline for processing data from  astronomical 
long-slit spectroscopy observations with CCD detectors.

The pipeine is designed to aim for **simplicity**, **manual execution**, **transparency** and **robustness**. The inspiration for the pipeline is to provide a manual counterpart to the 
well-established semi-automated and aumated pipelines. The intented use-cases are **teaching** and **edge-case observations**, where 
automated pipelines fail due to very low signal-to-noise ratio, several objects being very close 
on the detector and alike. For further elaboration,
please see the [Statement of need](#statement-of-need). 

From raw data, the
pipeline can produce the following output:
- A calibrated 2D spectrum in counts and wavelength for every detector pixel.
- A 1D spectrum extracted from the 2D spectrum in counts per wavelength (for point-like objects).
- A flux-calibrated 1D spectrum in $\frac{\text{erg}}{\text{s} \cdot \text{cm}^2 \cdot \text{Å}}$ (for point-like objects).


The products are obtained by performing standard procedures for
detector calibrations [@handbook] [@Howell_2006], comsic-ray subtraction [@cr_1] [@cr_2]
, and 1D spectrum extraction [@Horne_1986] [@photutils].   



# Statement of need

# Pipeline

Test of references: 

[@pypeit:joss_pub]

# Limitations


# Acknowledgements


# References
