"""
Error logging utility for the application.
"""

import logging
import os
from datetime import datetime


def setup_logger(log_dir='logs', log_level=logging.INFO):
    """
    Set up application logger that writes to both file and console.
    
    Parameters
    ----------
    log_dir : str
        Directory to store log files
    log_level : int
        Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'profile_fitting_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger('ProfileFitting')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only warnings and errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("=" * 60)
    logger.info("Profile Fitting Application Started")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    return logger


# Global logger instance
_logger = None


def get_logger():
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log_error(message, exception=None):
    """
    Log an error message with optional exception details.
    
    Parameters
    ----------
    message : str
        Error message
    exception : Exception, optional
        Exception object to log
    """
    logger = get_logger()
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)


def log_warning(message):
    """Log a warning message."""
    get_logger().warning(message)


def log_info(message):
    """Log an info message."""
    get_logger().info(message)


def log_debug(message):
    """Log a debug message."""
    get_logger().debug(message)
