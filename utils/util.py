import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
try:
    import ruamel_yaml as yaml
except ImportError:
    import ruamel.yaml as yaml


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.safe_load(handle)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=4)

'''
def inf_loop(data_loader):
    # wrapper function for endless data loader. 
    for loader in repeat(data_loader):
        yield from loader
'''


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'cur_total', 'cur_counts'])
        self.keys = keys
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def reset_batch(self):
        self._data.cur_total.values[:] = 0
        self._data.cur_counts.values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.cur_total[key] += value * n
        self._data.cur_counts[key] += n

    def avg(self, key):
        return self._data.total[key] / self._data.counts[key]

    def avg_batch(self, key):
        return self._data.cur_total[key] / self._data.cur_counts[key]

    def result(self):
        return {k: self.avg(k) for k in self._data.index}
