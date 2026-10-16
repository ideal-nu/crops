"""Logger setup for cosserat rod simulation and optimization."""

import logging
import os
from datetime import datetime

# Global configuration
LOGGING_ENABLED = True
CONSOLE_LOGGING_ENABLED = True
LOG_LEVEL = logging.INFO
LOG_DIR_ROOT = "./crops/logs"

_loggers = {}
_pending_loggers = []


def setup_logging(run_name=None) -> str:
  """Sets up the logging directory structure and basic configuration.

  Returns the path to the current log directory.
  """
  if not LOGGING_ENABLED:
    return None

  # Create root logs directory if it doesn't exist
  if not os.path.exists(LOG_DIR_ROOT):
    os.makedirs(LOG_DIR_ROOT)

  # Create timestamped run directory
  timestamp = datetime.now().strftime("%d_%m_%Y_%H_%_M_%S")
  if run_name:
    folder_name = f"{timestamp}_{run_name}"
  else:
    folder_name = timestamp

  current_log_dir = os.path.join(LOG_DIR_ROOT, folder_name)
  os.makedirs(current_log_dir)
  print(f"Logging directory created at: {current_log_dir}")
  return current_log_dir


# Store the current log directory globally once setup is called
_current_log_dir = None


def init_logging(run_name=None):
  global _current_log_dir
  _current_log_dir = setup_logging(run_name)

  for logr in _loggers.values():
    logr.setLevel(LOG_LEVEL)
    for handler in logr.handlers:
      handler.setLevel(LOG_LEVEL)
      if isinstance(handler, logging.StreamHandler) and not CONSOLE_LOGGING_ENABLED:
        logr.removeHandler(handler)

  # Process any loggers that were waiting for the directory to be created
  if _current_log_dir and _pending_loggers:
    formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for name, filename in _pending_loggers:
      if name in _loggers:
        logger = _loggers[name]
        file_path = os.path.join(_current_log_dir, filename)
        fh = logging.FileHandler(file_path)
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Clear the pending list
    _pending_loggers.clear()
  return _current_log_dir


def get_logger(name, filename=None) -> logging.Logger:
  """Get a logger with a specific name.
  If filename is provided, logs for this logger will go to that file
  inside the current run directory. If not provided, only console logging is set up.
  """
  if not LOGGING_ENABLED:
    # Return a dummy logger that does nothing
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger

  if name in _loggers:  # check if logger already exists
    return _loggers[name]

  logger = logging.getLogger(name)
  logger.setLevel(LOG_LEVEL)

  # Avoid adding handlers multiple times
  if not logger.handlers:
    formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if CONSOLE_LOGGING_ENABLED:
      ch = logging.StreamHandler()
      ch.setLevel(LOG_LEVEL)
      ch.setFormatter(formatter)
      logger.addHandler(ch)

    # File Handler
    if filename:
      if _current_log_dir:
        # Directory exists, add handler immediately
        file_path = os.path.join(_current_log_dir, filename)
        fh = logging.FileHandler(file_path)
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
      else:
        # Directory doesn't exist yet, queue this logger for later
        _pending_loggers.append((name, filename))

  _loggers[name] = logger
  return logger