import numpy as np

def estimate_dark(frame, exptime):

    from pylongslit.parser import detector_params

    gain = detector_params["gain"] #e/ADU

    dark = detector_params["dark_current"] #e/s/pixel
    
    # in electrons
    dark_frame_e = dark * exptime
    dark_frame_counts = dark_frame_e / gain

    # poisson noise for darks, from 
    #     From:
    # Richard Berry, James Burnell - The Handbook of Astronomical Image Processing
    # -Willmann-Bell (2005), p. 45 - 46
    dark_noise_error = np.sqrt(dark_frame_counts)

    return dark_frame_counts, dark_noise_error