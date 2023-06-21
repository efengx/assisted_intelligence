import psutil

from pynvml import (
    nvmlInit, 
    nvmlDeviceGetCount, 
    nvmlSystemGetDriverVersion, 
    nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName, 
    nvmlDeviceGetHandleByIndex,
    NVMLError
)
from fengxai.utils.log_center import create_logger

logger = create_logger()

def print_gpu_utilization():
    nvmlInit()
    logger.info(f"Driver Version: {nvmlSystemGetDriverVersion()}")
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        logger.info(f"Device {i} : {nvmlDeviceGetName(handle)}")
        info = nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU memory total(总共): {info.total/1024/1024/1024} GB.")
        logger.info(f"GPU memory occupied(占用): {info.used/1024/1024/1024} GB.")
        logger.info(f"GPU memory free(可用): {info.free/1024/1024/1024} GB.")

def print_summary(result):
    logger.info(f"Time: {result.metrics['train_runtime']:.2f}")
    logger.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_source()

def print_cpu_utilization():
    info = psutil.virtual_memory()
    logger.info(f"CPU 个数: {psutil.cpu_count()}")
    logger.info(f"CPU memory total: {info.total/1024/1024/1024} GB.")
    logger.info(f"CPU memory use: {info.used/1024/1024/1024} GB.")
    logger.info(f"CPU memory free: {info.free/1024/1024/1024} GB.")
    logger.info(f"CPU memory use percent: {info.percent} %")

def print_source():
    """
    打印CPU和GPU使用情况
    """
    print_cpu_utilization()
    try:
        print_gpu_utilization()
    except NVMLError as error:
        logger.info(f"gpu调用异常，error msg={error}")

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

