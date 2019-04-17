import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)
stream_log = logging.StreamHandler()
stream_log.setLevel(level=logging.WARNING)
file_log = logging.FileHandler('train.log')
file_log.setLevel(level=logging.WARNING)
logger.addHandler(stream_log)
logger.addHandler(file_log)
