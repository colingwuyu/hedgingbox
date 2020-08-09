"""A simple CSV logger.
   Support logging continuation.
"""

import csv
import os
import time

from absl import logging

from acme.utils import paths
from acme.utils.loggers import base


class CSVLogger(base.Logger):
  """Standard CSV logger."""

  _open = open

  def __init__(self,
               directory: str = '~/acme',
               label: str = '',
               time_delta: float = 0.):
    directory = paths.process_path(directory, 'logs', label, add_uid=False)
    self._file_path = os.path.join(directory, 'logs.csv')
    logging.info('Logging to %s', self._file_path)
    self._time = time.time()
    self._time_delta = time_delta
    if os.path.exists(self._file_path):
        self._header_exists = True
    else:
        self._header_exists = False

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""

    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()
    if now - self._time < self._time_delta:
      return
    self._time = now

    # Append row to CSV.
    with self._open(self._file_path, mode='a') as f:
      data = base.to_numpy(data)
      keys = sorted(data.keys())
      writer = csv.DictWriter(f, fieldnames=keys)
      if not self._header_exists:
        # Only write the column headers once.
        writer.writeheader()
        self._header_exists = True
      writer.writerow(data)

  @property
  def file_path(self) -> str:
    return self._file_path

  def clear(self):
    if self._header_exists:
        os.remove(self._file_path)
        self._header_exists = False
