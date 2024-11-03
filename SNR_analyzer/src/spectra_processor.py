import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm_joblib import tqdm_joblib
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import os
from joblib import Parallel, delayed
import re
from spectra_plotter import plot_result, get_lc
from shared import log_error, pickle_file, result_folder_path


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


def extract_values(hdul):
    bandwidth = None
    center_freq = None
    spectra = None
    try:
        bandwidth = hdul[1].header["BANDWID"]
        center_freq = hdul[1].header["RESTFREQ"]
    except KeyError as e:
        print(f"Key error: {e} in file {hdul.filename()}")
        bandwidth = None
        center_freq = None
    except IndexError as e:
        print(f"Index error: {e} in file {hdul.filename()}")
        bandwidth = None
        center_freq = None
    try:
        data = hdul[1].data
        spectra = get_stokesI(data)
    except Exception as e:
        print(f"Error processing data in file {hdul.filename()}: {e}")
        spectra = None

    return spectra, bandwidth, center_freq


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


def get_boxcar_filtered(intensities, window_sizes):
    filtered_intensities_list = []

    for boxcar_width in window_sizes:
        filtered_intensities_list.append(uniform_filter1d(intensities, size=boxcar_width))

    return filtered_intensities_list


def get_snr(spectra, window_sizes, only_best_SNR, dt):
    SNR_width_filtered_intensities_list = []
    _, intensities = get_lc(spectra, dt)
    filtered_intensities_list = get_boxcar_filtered(intensities, window_sizes)
    for filtered_intensities, boxcar_width in zip(filtered_intensities_list, window_sizes):
        half_boxcar = boxcar_width // 2
        # Find the peak (maximum value) in the light curve
        peak_index = np.argmax(filtered_intensities)
        peak = filtered_intensities[peak_index]

        # Check if the peak_index is within the valid range considering the boxcar window
        if not (half_boxcar <= peak_index < len(filtered_intensities) - half_boxcar):
            SNR_width_filtered_intensities_list.append((0, boxcar_width, filtered_intensities))
            continue

        # Define the signal region around the peak
        signal_indices = np.arange(max(0, peak_index - half_boxcar),
                                   min(len(filtered_intensities), peak_index + half_boxcar + 1))
        signal_region = filtered_intensities[signal_indices]
        # Define the noise region (everything outside the signal region)
        noise_indices = np.setdiff1d(np.arange(len(filtered_intensities)), signal_indices)
        noise_region = filtered_intensities[noise_indices]
        # Calculate the mean of the signal region
        mean_signal = np.mean(signal_region)  # not used
        # Calculate the standard deviation of the noise region
        std_noise = np.std(noise_region)
        # Calculate the mean of the noise region (Baseline)
        mean_noise = np.mean(noise_region)
        # Calculate the signal-to-noise ratio
        SNR_width_filtered_intensities_list.append(
            ((peak - mean_noise) / std_noise, boxcar_width, filtered_intensities))

    if only_best_SNR:
        best_SNR_width_filtered_intensities = max(SNR_width_filtered_intensities_list, key=lambda x: x[0])
        return best_SNR_width_filtered_intensities

    return SNR_width_filtered_intensities_list


# Function to sort each row
def sort_row_by_boxcar_size(size_row, value_row):
    paired_row = list(zip(size_row, value_row))
    paired_row.sort(key=lambda x: x[0])
    sorted_size_row, sorted_value_row = zip(*paired_row)
    return list(sorted_size_row), list(sorted_value_row)


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


def _process_fits_file(file_path, window_sizes, dt, scantime, only_best_snr):
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
            spectra = remove_source(spectra, noiseCal=True)

            window_sizes = [int(0.1 // dt), int(0.2 // dt), int(0.3 // dt), int(0.4 // dt), int(0.5 // dt), int(0.6 // dt), int(0.7 // dt)]
            #for i in range(1, 10):
            #    window_size = int(i // dt)
             #   window_sizes.append(window_size)

            if window_sizes is not None:
                SNR_and_filtered_intensities_list = get_snr(spectra, window_sizes, only_best_snr, dt)
                r = fits_name, (spectra, scantime, freqs), SNR_and_filtered_intensities_list
                try:
                    plot_result(result_folder_path, r)
                except Exception as e:
                    log_error("Error", f'Skipping {fits_name}. Error while plotting: {str(e)}')
                
                return (SNR_and_filtered_intensities_list[0], SNR_and_filtered_intensities_list[1], fits_name)

            else:
                return fits_name, (spectra, scantime, freqs)
    except Exception as e:
        log_error("Error", f'Skipping {file_path}. Error while processing: {str(e)}')


def process_spectra(fitsRaw_folder_path, window_sizes=None, dt=0.015625, scantime=28, only_best_snr=True):
    excluded_dates = set()

    with open('ignoreDates', 'r') as f:
        for line in f:
            excluded_dates.add(line.strip())

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
            delayed(_process_fits_file)(file_path, window_sizes, dt, scantime, only_best_snr) for file_path in
            fits_files)

    print('Processing complete.')
    return result

# result:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies))
# result if window_sizes are given:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies), (best SNR, boxcar width, filtered intensities))
# result if window_sizes are given and only_best_snr is set False:
# (fits name, (spectra calibrated and with source removed, scantime, frequencies), (SNR list, boxcar width list, filtered intensities list)
