import time
from shared import log_error, error_counts, pickle_file, log_file
import spectra_processor as sp
import pickle


# Simulate a method with various exceptions
def main_method(fitsRaw_folder_path):
    try:
        return sp.process_spectra(fitsRaw_folder_path)
    except Exception as e:
        log_error("critical", f'Critical Error running spectra processor: {str(e)}')


if __name__ == "__main__":

    start_time = time.time()
    fitsRaw_folder_path = "data/TELAMON_Data/TELAMON_RawData/S20mm"
    snr_width_filename = None

    try:
        snr_width_filename = main_method(fitsRaw_folder_path)
    except Exception as e:
        log_error("critical", f'Unhandled exception: {str(e)}')

    end_time = time.time()
    execution_time = end_time - start_time

    # Header information to be added
    header_info = (
        f"Execution Time: {execution_time:.2f} seconds\n"
        f"Error Counts:\n"
        f"  INFO: {error_counts['info']}\n"
        f"  WARNING: {error_counts['warning']}\n"
        f"  ERROR: {error_counts['error']}\n"
        f"  CRITICAL: {error_counts['critical']}\n"
        f"Pickle File: {pickle_file}\n"
    )

    with open(pickle_file, 'wb') as file:
        pickle.dump(snr_width_filename, file)

    print(f'Successfully saved to {pickle_file}')

    # Read the original log file
    with open(log_file, 'r') as f:
        logs = f.read()

    # Overwrite the log file with the header info followed by the logs
    with open(log_file, 'w') as f:
        f.write(header_info + "\n" + logs)

    
