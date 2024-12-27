import os
import logging
from logging.handlers import RotatingFileHandler

from config import results_dir


def setup_logging(name):
    """
    Set up logging for the project.
    Creates a log file in the results directory and ensures
    log messages are properly formatted and rotated.
    
    Input:
    - name (str): The name to assign to the logger. This is typically the name of the module or application 
                  where the logger is being used. It helps identify log messages coming from different parts 
                  of the project.

    Output:
    - logger (logging.Logger): The configured logger object.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f"log.log")

    # Configure the root logger
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Add a rotating file handler to prevent excessive log file size
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Attach the handler to the logger
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    return logger
