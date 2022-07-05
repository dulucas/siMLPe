import logging
from utils.pyt_utils import ensure_dir

def print_and_log_info(logger, string):
    logger.info(string)

def get_logger(file_path, name='train'):
    log_dir = '/'.join(file_path.split('/')[:-1])
    ensure_dir(log_dir)

    logger = logging.getLogger(name)
    hdlr = logging.FileHandler(file_path, mode='a')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
