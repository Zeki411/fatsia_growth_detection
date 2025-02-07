import os
import sys
import logging
import platform

import warnings
import tqdm




# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html


# Other Constants
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
DEFAULT_CFG_PATH = ""  # default config path
LOGGING_NAME = ""
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

VERBOSE = str(os.getenv("LOG_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format

if TQDM_RICH := str(os.getenv("TQDM_RICH", False)).lower() == "true":
    from tqdm import rich


class TQDM(rich.tqdm if TQDM_RICH else tqdm.tqdm):
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options.

    Attributes:
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:
        __init__: Initializes the TQDM object with custom settings.

    Examples:
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a custom TQDM progress bar.

        This class extends the original tqdm class to provide customized behavior for Ultralytics projects.

        Args:
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.

        Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.

        Examples:
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):
            ...     # Your code here
            ...     pass
        """
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)  # suppress tqdm.rich warning
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)
        

# TODO: refactor the "ultralytic" in the future
def set_logging(name="LOGGING_NAME", verbose=True):
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger. Defaults to "LOGGING_NAME".
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise. Defaults to True.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string
