import logging
import os
from logging import Filter


class SingleLineFormatter(logging.Formatter):
    """
    Custom formatter to replace newlines in log messages with literal \n
    and indent subsequent lines to align with the beginning of the first line.
    """

    def format(self, record):
        # Call the default formatter to get the message
        original_message = super().format(record)

        # Determine the initial indentation based on the prefix (up to the message content)
        initial_indent = ' ' * (len(self.formatTime(record)) + 3 + 15 + 3 + len(record.levelname) + 3)  # Adjust width as per format

        # Replace all newlines with \n and indent subsequent lines
        formatted_message = original_message.replace('\n', f'\n{initial_indent}')
        return formatted_message


class NoHttpRequestFilter(Filter):
    """
    Custom filter to remove HTTP request logs from specific libraries.
    """

    def filter(self, record):
        # Filter out any log messages that contain 'HTTP Request'
        return 'HTTP Request' not in record.getMessage()


def setup_logger(log_filename):
    """
    Set up the logger to log messages to a file, ensuring all output is single line,
    includes the module name (justified), and suppresses unwanted HTTP request logs.

    Args:
        log_filename (str): Name of the log file (without the path).
    """
    # Ensure the logs directory exists

    log_dir = os.path.dirname(__file__) + "/../logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Full path to the log file
    log_file_path = os.path.join(log_dir, log_filename)

    # Set up basic logging configuration with a custom formatter
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set level to INFO
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf8')
    file_handler.setLevel(logging.INFO)

    # Use custom formatter for single-line output including module name
    # Fixed-width padding for module name (e.g., 15 characters wide)
    file_formatter = SingleLineFormatter('%(asctime)s - %(module)-15s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    # Console handler (optional, remove if not needed)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)  # Use the same formatter for console

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Filter to remove unwanted HTTP request logs
    logger.addFilter(NoHttpRequestFilter())

    # Suppress logs from libraries (openai, langchain, urllib3, and specific modules)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("_client").setLevel(logging.ERROR)  # Explicitly handle _client

    logger.info(f"Logger initialized and logging to {log_file_path}")