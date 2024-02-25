import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        if save_dir is None:
            config['handlers'].pop('info_file_handler')
            config['root']['handlers'].remove('info_file_handler')
        else:
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op