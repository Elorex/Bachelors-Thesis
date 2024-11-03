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


def plot_result(result_folder_path, *args, dt=0.015625, action="save"):
    if len(args) == 1 and isinstance(args[0], tuple):
        # Case 1: Single tuple
        fits_name, spectra_data, snr_data = args[0]
    elif len(args) == 3 and all(isinstance(arg, tuple) for arg in args):
        # Case 2: Three tuples (fits_name, spectra_data, snr_data)
        fits_name, spectra_data, snr_data = args
    elif len(args) == 7:
        # Case 3: Individual components
        fits_name = args[0]
        spectra_data = args[1:4]
        snr_data = args[4:7]
    else:
        raise ValueError("Invalid arguments. Pass either a single tuple, three tuples, or seven individual elements.")

    spectra, scantime, freqs = spectra_data
    snr, boxcar_width, filtered_intensities = snr_data

    # Your existing code to get the data and perform the calculations
    times, intensities = get_lc(spectra, dt)
    half_boxcar = boxcar_width // 2
    peak_index = np.argmax(filtered_intensities)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

    # Plot 1
    ax[0].plot(times, intensities, label='Original', color='lightgray')
    ax[0].plot(times, filtered_intensities, label='Filtered', linestyle='-')
    ax[0].axvline(times[peak_index], color='red', linestyle=':', label='Peak')
    if snr != 0:
        ax[0].axvline(times[peak_index - half_boxcar], color='black', linestyle='--', label='Signal region')
        ax[0].axvline(times[peak_index + half_boxcar], color='black', linestyle='--')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Counts')
    ax[0].legend()

    # Plot 2
    # +10 for better visibility against white background
    im = ax[1].imshow(spectra + 10, norm=colors.LogNorm(), extent=[np.min(freqs), np.max(freqs), scantime, 0],
                      aspect='auto')
    fig.colorbar(im, ax=ax[1])  # Add colorbar to the second plot
    ax[1].set_xlabel("Frequency [GHz]")
    ax[1].set_ylabel("Time [s]")

    x_min, x_max = ax[1].get_xlim()
    y_coord_1 = (scantime / len(spectra)) * (peak_index - half_boxcar)
    y_coord_2 = (scantime / len(spectra)) * peak_index
    y_coord_3 = (scantime / len(spectra)) * (peak_index + half_boxcar)

    ax[1].text(x_min, y_coord_1, '-', color='black', fontsize=20, ha='center', va='center')
    ax[1].text(x_min, y_coord_2, '-', color= 'red', fontsize=20, ha='center', va='center')
    ax[1].text(x_min, y_coord_3, '-', color='black', fontsize=20, ha='center', va='center')

    # Text to add above the plots
    text1 = f'Boxcar Width: {boxcar_width:.2f} --> {boxcar_width * dt:.4f}s'
    text2 = f'Signal-to-Noise Ratio (SNR): {snr:.2f}'

    # Add text above the plots
    fig.text(0.5, 0.95, text1, ha='center', fontsize=12)
    fig.text(0.5, 0.91, text2, ha='center', fontsize=12)

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
        snr_str = str(snr).replace('.', '-')
        # Save the plot as an image
        png_filename = fits_name.replace('.fits', '.png')
        png_filename = f'{result_folder_path}/SNR_plots/SNR{snr_str}_' + png_filename.split('/')[-1]
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', metadata=metadata)

        # Close the plot
        plt.close()
    else:
        print("not plotting")

    if action == "show":
        plt.show()
