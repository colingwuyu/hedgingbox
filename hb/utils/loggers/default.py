"""Default logger."""

from acme.utils.loggers import aggregators
from acme.utils.loggers import base
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from hb.utils.loggers import csv


def make_default_logger(
    directory: str = '~/acme',
    label: str,
    save_data: bool = True,
    time_delta: float = 1.0,
) -> base.Logger:
  """Make a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Ignored.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger (pipe) object that responds to logger.write(some_dict).
  """
  terminal_logger = terminal.TerminalLogger(label=label, time_delta=time_delta)

  loggers = [terminal_logger]
  if save_data:
    loggers.append(csv.CSVLogger(directory, label))

  logger = aggregators.Dispatcher(loggers)
  logger = filters.NoneFilter(logger)
  logger = filters.TimeFilter(logger, time_delta)
  return logger
