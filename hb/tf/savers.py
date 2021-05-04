import os
from typing import Mapping, Union, List
import shutil
import json

import sonnet as snt
from acme.utils import paths
from acme.tf.savers import make_snapshot
import tensorflow as tf
from zipfile import ZipFile


class Snapshotter:
    """
    Objects which can be snapshotted are limited to Sonnet or tensorflow Modules
    which implement a a __call__ method. This will save the module's graph and
    variables such that they can be loaded later using `tf.saved_model.load`. See
    https://www.tensorflow.org/guide/saved_model for more details.

    The Snapshotter is typically used to save infrequent permanent self-contained
    snapshots which can be loaded later for inspection. For frequent saving of
    model parameters in order to guard against pre-emption of the learning process
    see the Checkpointer class.

    Usage example:

    ```python
    model = snt.Linear(10)
    snapshotter = Snapshotter(objects_to_save={'model': model})

    for _ in range(100):
      # ...
      snapshotter.save()
    ```
    """

    def __init__(
        self, bot_name: str, bot_json: dict,
        objects_to_save: Mapping[str, snt.Module],
        *,
        directory: str = '~/acme/',
    ):
        """Builds the saver object.

        Args:
          objects_to_save: Mapping specifying what to snapshot.
          directory: Which directory to put the snapshot in.
        """
        objects_to_save = objects_to_save or {}
        self._snapshots = {}

        # Save the base directory path so we can refer to it if needed.
        self.pardir = directory
        self.directory = paths.process_path(
            directory, 'snapshots', add_uid=False)
        self.name = bot_name
        self.json = bot_json

        # Save a dictionary mapping paths to snapshot capable models.
        for name, module in objects_to_save.items():
            path = os.path.join(self.directory, name)
            self._snapshots[path] = make_snapshot(module)

    def save(self):
        """Snapshots if it's the appropriate time, otherwise no-ops.

        Returns:
          A boolean indicating if a save event happened.
        """
        if self._snapshots:
            # Save any snapshots.
            for path, snapshot in self._snapshots.items():
                tf.saved_model.save(snapshot, path)
            with open(os.path.join(self.directory, 'desc'), 'w') as desc_file:
                desc_file.write(json.dumps(self.json))
            if os.path.exists(os.path.join(self.pardir, self.name+'.bot')):
                os.remove(os.path.join(self.pardir, self.name+'.bot'))
            shutil.make_archive(
                os.path.join(self.pardir, self.name), 'zip', self.directory)
            os.rename(os.path.join(self.pardir, self.name+'.zip'),
                      os.path.join(self.pardir, self.name+'.bot'))
            shutil.rmtree(self.directory)


def load_bot(bot_file: str):
    """Load snapshotted functions

    Args:
        file (str): snapshot directory. Defaults to '~/acme'.
    """
    directory = os.path.dirname(bot_file)
    tmp_dir = paths.process_path(
        directory+'/temp/', add_uid=True)

    with ZipFile(bot_file, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(tmp_dir)
    ret_functions = {}
    for function_name in ('policy', 'critic'):
        ret_functions[function_name] = tf.saved_model.load(
            os.path.join(tmp_dir, function_name))
    bot_json = open(os.path.join(tmp_dir, 'desc'))
    bot_dict = json.load(bot_json)
    bot_json.close()
    shutil.rmtree(directory+'/temp/')
    return ret_functions, bot_dict
