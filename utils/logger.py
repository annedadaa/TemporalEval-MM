import logging
import os

# ANSI escape codes
RESET = "\033[0m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RED = "\033[91m"

class ColorFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: BLUE + "[DEBUG] %(message)s" + RESET,
        logging.INFO: GREEN + "[INFO] %(message)s" + RESET,
        logging.WARNING: YELLOW + "[WARN] %(message)s" + RESET,
        logging.ERROR: MAGENTA + "[ERROR] %(message)s" + RESET,
        logging.CRITICAL: RED + "[CRITICAL] %(message)s" + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._style._fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name: str = "app_logger", level=logging.INFO, log_file_path: str = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        # Console handler with color
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(ColorFormatter())
        logger.addHandler(stream_handler)

        # Optional file handler (without ANSI colors)
        if log_file_path:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger
