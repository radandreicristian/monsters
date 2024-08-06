import logging
import logging.config

import json


def get_logger(config_path: str = "src/config/logging_config.json") -> logging.Logger:
    logging.config.dictConfig(json.load(open(config_path)))
    return logging.getLogger()
