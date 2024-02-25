__all__ = ['BiDict']


class BiDict(object):
    def __init__(self, d: dict = {}):
        self._data = {}
        self._data_t = {}
        for key, value in d.items():
            self._data[key] = value
            self._data_t[value] = key

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert all((
                item.start is None,
                item.stop is not None,
                item.step is None
            ))
            return self._data_t[item.stop]
        return self._data[item]
