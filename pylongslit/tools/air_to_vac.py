"""
A tool for converting air wavelengths to vacuum wavelengths.
Expected input file format:
Wavelength in air (Angstroms) Ion Name (optional) 
"""

import numpy as np
import argparse
from PyAstronomy import pyasl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert air wavelengths to vacuum wavelengths."
    )
    parser.add_argument(
        "infile", type=str, help="Input file containing wavelengths in air."
        "Expected format: Wavelength in air (Angstroms), Ion Name (optional)"
    )
    parser.add_argument(
        "outfile", type=str, help="Output file containing wavelengths in vacuum."
    )
    args = parser.parse_args()

    # Read in the input file
    data = np.genfromtxt(args.infile, dtype=None, encoding="utf-8")

    # Convert air wavelengths to vacuum wavelengths
    vac_wavelengths = []
    ions = []
    for row in data:
        air_wavelength = row[0]
        ion = row[1] if len(row) > 1 else ""
        vac_wavelength = pyasl.airtovac2(air_wavelength)
        vac_wavelengths.append(vac_wavelength)
        ions.append(ion)
        print(
            f"{ion}. Air: {air_wavelength} Angstroms -> Vacuum: {vac_wavelength:.6f} Angstroms"
        )

    # Write the vacuum wavelengths to the output file
    np.savetxt(args.outfile, np.column_stack((vac_wavelengths, ions)), fmt="%s")
    print(f"Wrote vacuum wavelengths to {args.outfile}")
