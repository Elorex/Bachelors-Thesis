import logging
from collections import defaultdict


result_folder_path = "data/TELAMON_Data/Results/RUN3_5"

log_file = (f"{result_folder_path}/app.log")
pickle_file =(f"{result_folder_path}/app.pickle")

# Configure the logging system
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()]
)

# Create a logger for the shared module
logger = logging.getLogger(__name__)

    
# Dictionary to track error counts
error_counts = defaultdict(int)


# Function to log errors and update counts
def log_error(error_type, message, exc_info=False):
    if error_type == "info":
        logger.info(message, exc_info=exc_info)
    elif error_type == "warning":
        logger.warning(message, exc_info=exc_info)
    elif error_type == "error":
        logger.error(message, exc_info=exc_info)
    elif error_type == "critical":
        logger.critical(message, exc_info=exc_info)
    error_counts[error_type] += 1