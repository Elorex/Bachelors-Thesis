import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm_joblib import tqdm_joblib
from scipy.optimize import curve_fit
import scipy.ndimage
import os
from joblib import Parallel, delayed
import re
from spectra_plotter import plot_result, get_lc
from shared import log_error, pickle_file, result_folder_path
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage.morphology import dilation, erosion, disk
import cv2



class CurveFitException(Exception):
    pass


def _is_excluded(filename, excluded_dates, partial_exclusions=None):
    if partial_exclusions is None:
        partial_exclusions = {}
    try:
        # Extract the date and wavelength from the filename
        date_part = filename.split('_')[1]
        wavelength_match = re.search(r'(S\d+mm)', filename)
        wavelength = wavelength_match.group(1) if wavelength_match else None

        if date_part < '2021-09-19':
            return True
        if date_part in excluded_dates:
            return True
        if date_part in partial_exclusions:
            if wavelength and wavelength not in partial_exclusions[date_part]:
                return True
    except IndexError:
        return True  # Exclude files that don't match the expected format
    except AttributeError:
        return True  # Exclude files that don't have a proper wavelength format

    return False


def get_stokesI(data):
    spectra = []
    mjds = []
    phi = 0  # reference angle for Q-U reference frame

    # we need to convert the data to total intensity
    # since the data are recorded in circular polarization
    for i in range(len(data)):
        spectrum = []

        mjds.append(data[i][0])

        # extract polarization parameters
        lcp = np.array(data[i][1][0])
        rcp = np.array(data[i][1][1])
        COS = np.array(data[i][1][2])
        SIN = np.array(data[i][1][3])

        I = lcp + rcp
        Q = np.cos(phi) * COS - np.sin(phi) * SIN
        U = np.sin(phi) * COS + np.cos(phi) * SIN
        V = lcp - rcp

        spectra.append(I)  # append STOKES I (Q=1,U=2,V=3)

    return np.array(spectra)


# Define the Gaussian function
def _gauss(x, A, B, C, D):
    y = A * np.exp(-1 * B * (x - C) ** 2) + D
    return y


# does a simple bandpass calibration by dividing all data by the average spectrum
def calibrate_bandpass(spectra):
    for i in range(len(spectra[0])):
        add_count = 0
        for j in range(len(spectra)):
            add_count += spectra[j][i]
        corr_factor = add_count / len(spectra)
        for j in range(len(spectra)):
            spectra[j][i] = spectra[j][i] / corr_factor

    return spectra


def remove_noise_cal(spectra, isLC=False, removeSource=False, showPlots=False):
    # remove noise cal effect
    for j in range(len(spectra[0])):

        # seperate the two/four phases
        bin1 = []
        bin2 = []
        count = 2
        switch = True
        for i in range(len(spectra)):
            if count == 0:
                switch = not switch
                count = 2
            if switch:
                bin1.append(spectra[i][j])
            elif not switch:
                bin2.append(spectra[i][j])
            count -= 1

        fit1 = np.full(len(bin1), np.average(bin1))
        fit2 = np.full(len(bin2), np.average(bin2))

        bin1 = bin1 - fit1
        bin2 = bin2 - fit2

        final_time = []
        i = 0
        while i < len(bin1):
            final_time.append(bin1[i])
            if not i + 1 >= len(bin1):
                final_time.append(bin1[i + 1])
            if not i >= len(bin2):
                final_time.append(bin2[i])
            if not i + 1 >= len(bin2):
                final_time.append(bin2[i + 1])
            i += 2

        for i in range(len(spectra)):
            spectra[i][j] = final_time[i]
    return spectra


def remove_source(spectra, isLC=False, noiseCal=False, showPlots=False):
    if noiseCal:
        # remove noise cal effect
        for j in range(len(spectra[0])):

            # seperate the two/four phases
            bin1 = []
            bin2 = []
            count = 2
            switch = True
            for i in range(len(spectra)):
                if count == 0:
                    switch = not switch
                    count = 2
                if switch:
                    bin1.append(spectra[i][j])
                elif not switch:
                    bin2.append(spectra[i][j])
                count -= 1

            fit1 = np.full(len(bin1), np.average(bin1))
            fit2 = np.full(len(bin2), np.average(bin2))

            bin1 = bin1 - fit1
            bin2 = bin2 - fit2

            final_time = []
            i = 0
            while i < len(bin1):
                final_time.append(bin1[i])
                if not i + 1 >= len(bin1):
                    final_time.append(bin1[i + 1])
                if not i >= len(bin2):
                    final_time.append(bin2[i])
                if not i + 1 >= len(bin2):
                    final_time.append(bin2[i + 1])
                i += 2

            for i in range(len(spectra)):
                spectra[i][j] = final_time[i]

    # remove source
    if isLC:
        averaged_spectrum = spectra
    else:
        averaged_spectrum = np.mean(spectra, axis=1)

    try:
        parameters, covariance = curve_fit(
            _gauss,
            np.arange(len(averaged_spectrum)),
            np.array(averaged_spectrum),
            p0=np.asarray([0.1, 1 / 100, len(averaged_spectrum) / 2, np.average(averaged_spectrum)]),
            maxfev=2000,
            bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, len(averaged_spectrum), np.inf])
        )
    except Exception as e:
        parameters = [0, 0, 0, 0]
        raise CurveFitException(f"Curve fitting failed: {e}")

    fit_A1, fit_B1, fit_C1, fit_D1 = parameters

    fit = _gauss(np.arange(len(averaged_spectrum)), fit_A1, fit_B1, fit_C1, fit_D1)

    if showPlots:
        plt.plot(np.arange(len(averaged_spectrum)), averaged_spectrum, color="lightgray", label="Averaged Spectrum")
        plt.plot(np.arange(len(fit)), fit, color="blue", label="Gaussian Fit")
        plt.legend()
        plt.show()

    spectra = (spectra.T - fit).T

    return spectra


# Define the Gaussian function with noise
def _gauss_with_noise(x, A, B, C, D):
    y = A * np.exp(-1 * B * (x - C) ** 2) + D

    # Identify the indices within the two sigma region
    two_sigma_indices = np.where(np.abs(x - C) <= 2 * B)[0]

    # Add noise within the two sigma region
    noise_std_dev = 0.2 * A  # Standard deviation of the noise
    noise = np.random.normal(0, noise_std_dev, len(two_sigma_indices))
    y[two_sigma_indices] += noise

    return y


def add_rfi(spectra, width=None, length=None):
    if length is None:
        length = [1, 100]
    if width is None:
        width = [1, 10]
    width = np.random.randint(*width)
    length = np.random.randint(*length)
    intensity = np.max(spectra) * 2
    position_width = np.random.randint(1, len(spectra[0]) - width - 1)
    position_length = np.random.randint(1, len(spectra) - 1)
    print(position_width)
    print(position_length)
    x = np.arange(len(spectra))
    box = _gauss_with_noise(x, intensity, 1 / length, position_length / 2, 0)

    for i in range(position_width, position_width + width):
        spectra[:, i] += box
    return spectra

def laplace_conv(spectra):
     # Apply a Gaussian filter to smooth the data
    smoothed_spectra = scipy.ndimage.gaussian_filter(spectra, sigma=1)

    # Define a 2D Laplacian kernel
    laplacian_kernel = np.array([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]])
    #laplacian_kernel = np.array([[-1,  -1, -1],
    #                            [-1, 8, -1],
    #                            [-1,  -1, -1]])

    # Apply the Laplacian convolution to the smoothed data
    filtered_spectra = scipy.ndimage.convolve(smoothed_spectra, laplacian_kernel)

    # Calculate the threshold value for anomaly detection
    threshold = np.mean(filtered_spectra) + 2 * np.std(filtered_spectra)

    # Create a binary mask of anomalies
    anomalies = np.abs(filtered_spectra) > threshold

    return anomalies

def nlm_canny(spectra, h=10, templateWindowSize=5, searchWindowSize=15, low_threshold=50, high_threshold=150):
    # Step 1: Normalize the spectrogram to the range [0, 255]
    spectra_normalized = cv2.normalize(spectra, None, 0, 255, cv2.NORM_MINMAX)
    spectra_normalized = np.uint8(spectra_normalized)
    
    # Step 2: Apply Non-Local Means Denoising
    denoised_spectra = cv2.fastNlMeansDenoising(spectra_normalized, None, h, templateWindowSize, searchWindowSize)
    
    # Step 3: Apply Canny Edge Detection
    edges = cv2.Canny(denoised_spectra, low_threshold, high_threshold)
    
    return edges



def _process_fits_file(file_path, scantime, removeSource):
    try:
        with fits.open(file_path) as hdul:
            fits_name = hdul.filename()
            bandwidth = hdul[1].header["BANDWID"]
            center_freq = hdul[1].header["RESTFREQ"]
            low_freq = center_freq - bandwidth / 2
            high_freq = center_freq + bandwidth / 2

            # and read out the data
            data = hdul[1].data
            # convert data from circular Polarization to StokesI
            spectra = get_stokesI(data)

            # extract some information
            freqs = np.linspace(low_freq / 1e9, high_freq / 1e9, len(spectra[0]))

            spectra = calibrate_bandpass(spectra)
            spectra = remove_noise_cal(spectra)

            if removeSource:
                spectra = remove_source(spectra, noiseCal=True)

            anomalies = laplace_conv(spectra)
            #anomalies = nlm_canny(spectra)

            spectra_data = spectra, scantime, freqs


            ass = plot_result(result_folder_path, fits_name, spectra_data, anomalies)

            return (ass, fits_name)


           

    except Exception as e:
        log_error("Error", f'Skipping {file_path}. Error while processing: {str(e)}')


def process_spectra(fitsRaw_folder_path, scantime=28, removeSource = True):
    excluded_dates = set()
    
    with open('ignoreDates', 'r') as f:
        for line in f:
            excluded_dates.add(line.strip())
    print("hi1")
    # Create a list of all FITS file paths in the specified directory, excluding unwanted dates and wavelengths
    fits_files = [
        os.path.join(fitsRaw_folder_path, filename)
        for filename in os.listdir(fitsRaw_folder_path)
        if filename.endswith('.fits') and not _is_excluded(filename, excluded_dates)
    ]
    # Use joblib.Parallel with tqdm to process files in parallel with a progress bar
    num_jobs = -1  # Number of jobs to run in parallel, -1 means using all processors
    
    with tqdm_joblib(desc="Processing FITS files", total=len(fits_files)):
        result = Parallel(n_jobs=num_jobs)(
            delayed(_process_fits_file)(file_path, scantime, removeSource) for file_path in
            fits_files)

    print('Processing complete.')
    return result

# result:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies))
# result if window_sizes are given:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies), (best SNR, boxcar width, filtered intensities))
# result if window_sizes are given and only_best_snr is set False:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies), (SNR list, boxcar width list, filtered intensities list)
