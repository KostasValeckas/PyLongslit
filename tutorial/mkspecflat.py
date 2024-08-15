import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, bias_params, flat_params, output_dir
from utils import FileList, check_dimensions, show_overscan, open_fits, write_to_fits

"""
Module for creating a master flat framw from raw flat frames.
"""

def run_flats():

   # Extract the detector parameters
   xsize = detector_params["xsize"]
   ysize = detector_params["ysize"]

   overscan_x_start = detector_params["overscan_x_start"]
   overscan_x_end = detector_params["overscan_x_end"]
   overscan_y_start = detector_params["overscan_y_start"]
   overscan_y_end = detector_params["overscan_y_end"]

   # TODO: specify what direction is the spectral direction
   logger.info("Flat-field procedure running...")
   logger.info("Using the following parameters:")
   logger.info(f"xsize = {xsize}")
   logger.info(f"ysize = {ysize}")


   # read the names of the flat files from the directory
   file_list = FileList(flat_params["flat_dir"])

   if overscan_x_end != 0 and overscan_y_end != 0:
       logger.info("Non - zero overscan region is defined.")
       logger.info(
           f"Overscan region: x: {overscan_x_start}:{overscan_x_end},"
           f" y: {overscan_y_start}:{overscan_y_end}"
       )
       # Show the overscan region on a flat fram for Quality Assesment
       show_overscan()

   logger.info(f"Found {file_list.num_files} flat frames.")
   logger.info(f"Files used for flat-fielding:")
   print("------------------------------------")
   for file in file_list:
       print(file)
   print("------------------------------------")

   # Check if all files have the wanted dimensions
   # Will exit if they don't
   check_dimensions(file_list, xsize, ysize)

   # initialize a big array to hold all the flat frames for stacking
   bigflat = numpy.zeros((file_list.num_files, ysize, xsize), float)

   logger.info("Fetching the master bias frame...")

   try:
      BIASframe = fits.open(output_dir + '/master_bias.fits')
   except FileNotFoundError:
      logger.critical("Master bias frame not found.")
      logger.error("Make sure a master bias frame exists before proceeding with flats.")
      logger.error("Run the mkspecbias.py script first.")
      exit()

   BIAS = numpy.array(BIASframe[0].data)
   logger.info("Master bias frame found and loaded.")

   # loop over all the falt files, subtract the median value of the overscan,
   # subtract bias and stack them in the bigflat array
   for i, file in enumerate(file_list):
   
      rawflat = open_fits(flat_params["flat_dir"], file)

      logger.info(f"Processing file: {file}")

      data = numpy.array(rawflat[1].data)

      # TODO: if this is needed more - move it to utils
      if overscan_x_end != 0 and overscan_y_end != 0:
          overscan_mean = numpy.mean(
              data[overscan_y_start:overscan_y_end, overscan_x_start:overscan_y_end]
          )
          data = data - overscan_mean
          logger.info(
              f"Subtracted the median value of the overscan : {overscan_mean}"
          )

      data = data - BIAS
      logger.info("Subtracted the bias.")
      
      bigflat[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]
      
      norm = numpy.median(bigflat[i,100:1300, 100:400])

      logger.info(f"Normalising frame with the median of the frame :{norm}\n")
      bigflat[i,:,:] = bigflat[i,:,:]/norm

      # close the file handler
      rawflat.close()
   
   logger.info("Normalizing the final master flat-field....")

   # Calculate flat is median at each pixel
   medianflat = numpy.median(bigflat,axis=0)

   # Find a mean in spectral direction for each row
   lampspec = numpy.mean(medianflat,axis=1)

   for i in range(0,xsize-1):
      medianflat[:,i] = medianflat[:,i] / lampspec[:]

   logger.info("Flat frames processed.")
   logger.info("Attaching header and writing to disc...")

   #Write out result to fitsfile
   hdr = rawflat[0].header
   
   write_to_fits(medianflat, hdr, output_dir + "master_flat.fits", output_dir)

   logger.info(
       f"Master flat frame written to disc in {output_dir}, filename master_flat.fits"
   )

if __name__ == "__main__":
   run_flats()

