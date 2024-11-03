from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
import re

plt.switch_backend('Agg')


def get_lc(spectra, dt, periodogram=False):
    times = np.linspace(0, (len(spectra) - 1) * dt, len(spectra))
    intensities = np.nanmean(spectra, axis=1)

    if periodogram:
        freq, p_ls = LombScargle(times, np.nan_to_num(intensities)).autopower(minimum_frequency=0.000000001,
                                                                              maximum_frequency=10,
                                                                              normalization='psd',
                                                                              samples_per_peak=50)
        return freq, p_ls

    return times, intensities


def plot_lc(spectra, dt=0.015625, periodogram=False, doScatter=False, plotlim=None):
    if plotlim is None:
        plotlim = []
    if len(plotlim) > 0:
        ind_min = int(plotlim[0] / dt)
        ind_max = int(plotlim[1] / dt)
    else:
        ind_min = 0
        ind_max = len(spectra) - 1

    times = np.linspace(0, (len(spectra) - 1) * dt, len(spectra))
    times = times[ind_min:ind_max]
    intensities = np.nanmean(spectra, axis=1)
    intensities = intensities[ind_min:ind_max]
    print(times)
    print(len(spectra))
    if doScatter:
        plt.scatter(times, intensities)
    else:
        plt.plot(times, intensities)
    plt.xlabel("Time [s]")
    plt.ylabel("Counts")
    plt.show()

    if periodogram:
        freq, p_ls = LombScargle(times, np.nan_to_num(intensities)).autopower(minimum_frequency=0.000000001,
                                                                              maximum_frequency=10,
                                                                              normalization='psd',
                                                                              samples_per_peak=50)
        plt.plot(freq, p_ls)
        plt.yscale("log")
        plt.xlabel("Frequency")
        plt.ylabel("Spectral Power")
        plt.show()

        return freq, p_ls

    return times, intensities


def plot_spectrum(spectra, freqs):
    plt.plot(freqs, np.average(spectra, axis=0))
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Counts")
    plt.show()


def plot_waterfall(spectra, scantime, freqs, masked=False):
    if masked:
        mask = np.zeros(spectra.shape)
        mask[:, len(spectra[1]) // 2 - 2:len(spectra[1]) // 2 + 2] = 1
        spectra = np.ma.masked_array(spectra, mask)

    plt.imshow(spectra, norm=colors.LogNorm(), extent=[np.min(freqs), np.max(freqs), scantime, 0], aspect="auto")
    plt.colorbar()
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Time [s]")
    plt.show()


def get_ass(spectra, anomalies):
    # Count rows that have at least one True value
    rows_with_anomalies = np.any(anomalies, axis=1)
    num_rows_with_anomalies = np.sum(rows_with_anomalies)

    # Count columns that have at least one True value
    columns_with_anomalies = np.any(anomalies, axis=0)
    num_columns_with_anomalies = np.sum(columns_with_anomalies)

    # Calculate the effective size considering only rows and columns with anomalies
    effective_size = num_rows_with_anomalies * num_columns_with_anomalies

    try:
        effective_density = np.sum(anomalies) / effective_size
    except Exception as e:
        effective_density = 0

    pai = np.max(np.abs(spectra[anomalies]))
    mean_intensity = np.mean(np.abs(spectra[anomalies]))

     # Text to add above the plots
    text1 = f'Peak Anomaly Intensity: {pai:.2f}'
    text2 = f'Mean Intensity of all Anomalies: {mean_intensity:.2f}'
    text3 = f'Effective Anomaly Density: {effective_density:.3f}'

    ass = pai * effective_density # Anomaly Significance Score

    return ass, text1, text2, text3


def get_mmr(spectra,anomalies):
    mean_intensity = np.mean(np.abs(spectra[anomalies]))
    pai = np.max(np.abs(spectra[anomalies]))

    try:
        mmr = pai / mean_intensity
    except Exception as e:
        mmr = 0

    text1 = f'Peak Anomaly Intensity: {pai:.2f}'
    text2 = f'Sum of all Anomalies: {np.sum(anomalies)}'
    text3 = f'Mean Intensity of all Anomalies: {mean_intensity:.2f}'

    return mmr, text1, text2, text3


def get_snr(spectra,anomalies):
    intensity_std = np.std(spectra)
    pai = np.max(np.abs(spectra[anomalies]))

    try:
        snr = pai / intensity_std
    except Exception as e:
        snr = 0

    text1 = f'Peak Anomaly Intensity: {pai:.2f}'
    text2 = f'Sum of all Anomalies: {np.sum(anomalies)}'
    text3 = f'Standard Deviation: {intensity_std:.2f}'

    return snr, text1, text2, text3


def get_fas(spectra, anomalies):
    # Set the threshold based on the mean noise level
    mean_noise_level = 0.1
    threshold = mean_noise_level

    # Create a new mask with points above the threshold
    filtered_anomalies = (spectra > threshold) & anomalies
    # Calculate the Filtered Anomaly Sum (FAS)
    if np.any(filtered_anomalies):
        filtered_anomaly_data = spectra[filtered_anomalies]
        filtered_anomaly_sum = np.sum(filtered_anomaly_data)
    else:
        filtered_anomaly_sum = 0

    pai = np.max(np.abs(spectra[filtered_anomalies]))

    text1 = f'Peak Anomaly Intensity in Filtered Anomalies: {pai:.2f}'
    text2 = f'Sum of all Anomalies: {np.sum(spectra[anomalies]):.2f}'
    text3 = f'Filtered Anomaly Sum: {filtered_anomaly_sum:.2f}'

    return filtered_anomalies, filtered_anomaly_sum, text1, text2, text3


def get_pai(spectra, anomalies):
    pai = np.max(np.abs(spectra[anomalies]))

     # Text to add above the plots
    text1 = f'Peak Anomaly Intensity: {pai:.2f}'
    text2 = f'Number of Edges: {np.sum(anomalies)}'
    text3 = f'Mean Anomaly: {np.mean(np.abs(spectra[anomalies])):.2f}'

    return pai, text1, text2, text3



def plot_result(result_folder_path, fits_name, spectra_data, anomalies, dt=0.015625, action="save"):
    
    spectra, scantime, freqs = spectra_data

    #ass, text1, text2, text3 = get_ass(spectra, anomalies)
    mmr, text1, text2, text3 = get_mmr(spectra, anomalies)
    #snr, text1, text2, text3 = get_snr(spectra, anomalies)
    #anomalies, fas, text1, text2, text3 = get_fas(spectra, anomalies)
    #pai, text1, text2, text3 = get_pai(spectra, anomalies)


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

    # Plot 1
    ax[0].imshow(anomalies, cmap='hot', extent=[np.min(freqs), np.max(freqs), scantime, 0],aspect='auto')
    ax[0].set_xlabel("Frequency [GHz]")
    ax[0].set_ylabel("Time [s]")

    # Plot 2
    # +10 for better visibility against white background
    im = ax[1].imshow(spectra + 10, norm=colors.LogNorm(), extent=[np.min(freqs), np.max(freqs), scantime, 0],
                      aspect='auto')
    fig.colorbar(im, ax=ax[1])  # Add colorbar to the second plot
    ax[1].set_xlabel("Frequency [GHz]")
    ax[1].set_ylabel("Time [s]")
   

    # Add text above the plots
    fig.text(0.5, 0.95, text1, ha='center', fontsize=12)
    fig.text(0.5, 0.91, text2, ha='center', fontsize=12)
    fig.text(0.5, 0.87, text3, ha='center', fontsize=12)

    # Adjust the layout to make room for the text at the top
    plt.subplots_adjust(top=0.85)

    if action == "save":
        match = re.search(r'\d{4}-\d{2}-\d{2}', fits_name)
        if match:
            creation_date = match.group()
        else:
            raise ValueError("Creation date not found in filename.")
        metadata = {
            'CreationDate': creation_date
        }
        
        mmr_str = str(mmr*100).replace('.', '-') 
        # Save the plot as an image
        png_filename = fits_name.replace('.fits', '.png')
        png_filename = f'{result_folder_path}/Laplace_plots/MMR{mmr_str}_' + png_filename.split('/')[-1]
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', metadata=metadata)

        # Close the plot
        plt.close()

        return mmr
    else:
        print("not plotting")

    if action == "show":
        plt.show()

