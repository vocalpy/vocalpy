import logging
from rich.logging import RichHandler

logger = logging.getLogger("seqanalysis")


def config_logging():
    if logger.hasHandlers():
        logger.handlers.clear()

    stream_handler = RichHandler()

    logger.setLevel(level="INFO")
    stream_handler.setLevel(level="INFO")
    logger.setLevel(level="DEBUG")
    stream_handler.setLevel(level="DEBUG")

    fmt_shell = "%(message)s"

    shell_formatter = logging.Formatter(fmt_shell)

    # here we hook everything together
    stream_handler.setFormatter(shell_formatter)

    logger.addHandler(stream_handler)

    return logger
