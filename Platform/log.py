import logging

fl_logger = logging.getLogger('FL')
fl_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('stat/exp/fl.log')
fh.setLevel(logging.DEBUG)
fl_logger.addHandler(fh)


def logging_print(x):
    fl_logger.info(x)
    print(x)
