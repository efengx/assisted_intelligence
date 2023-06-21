import os
import wandb

from fengxai.inference.main_inference import main_inference
from fengxai.utils.log_center import create_logger
from fengxai.config.inference_config import InferenceArguments
from transformers import HfArgumentParser

logger = create_logger()

# 设置环境变量：1开启离线训练
# 离线的 model 
# os.environ["TRANSFORMERS_OFFLINE"]="1"
# 离线的 datasets
# os.environ['HF_DATASETS_OFFLINE']="1"

# wandb 环境变量
os.environ['WANDB_API_KEY']="93523e57b94611e1a558a6541f834f17dd400be5"
os.environ['WANDB_PROJECT']="assisted intelligence"

def auto_dataset():
    logger.info(f"step-notify: 告诉wandb根据环境变量信息开始{os.getenv('WANDB_PROJECT')}任务")
    wandb.init(project="rjxai-inference", entity="rjxai")

    logger.info("step-init: 加载参数")
    parser = HfArgumentParser((InferenceArguments))

    inference_args = parser.parse_args()
    logger.info(inference_args)

    main_inference(
        inference_args=inference_args,
    )
    
if __name__ == "__main__":
    auto_dataset()
