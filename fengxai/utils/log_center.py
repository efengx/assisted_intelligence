import multiprocessing
import logging
import sys
import os

os.environ['WANDB_API_KEY']="93523e57b94611e1a558a6541f834f17dd400be5"

def create_logger():
    """
    防止日志重复打印
    """
    logger = multiprocessing.get_logger()
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    # %(thread)d 线程id
    # %(threadName)s 线程名称
    # %(process)d 进程id
    formatter = logging.Formatter('%(asctime)s,%(msecs)03d %(levelname)-8s %(thread)d %(threadName)s %(process)d [%(filename)s:%(lineno)d] %(message)s')
    # handler = logging.FileHandler('logs/your_file_name.log')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not logger.hasHandlers():
        logger.info(f"hashandlers: {logger.hasHandlers}")
        logger.addHandler(handler)
    return logger
