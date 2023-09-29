import logging
from colorama import init, Fore, Style

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            record.levelname = f"{Fore.CYAN}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.propagate = False

if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG)
    colored_formatter = ColoredFormatter('%(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

# Set third-party logger levels here
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("jaxlib").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("modal-utils").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
