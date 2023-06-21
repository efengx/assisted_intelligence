from fengxai.utils.log_center import create_logger
from fengxai.config.clean_dataset_config import CleanDatasetArguments
from fengxai.dataset.clean_dataset_factory import CleanDatasetFactory
from transformers import HfArgumentParser

logger = create_logger()

# 设置环境变量：1开启离线训练
# 离线的 model 
# os.environ["TRANSFORMERS_OFFLINE"]="1"
# 离线的 datasets
# os.environ['HF_DATASETS_OFFLINE']="1"

def auto_dataset():
    logger.info("step-init: 加载参数")
    parser = HfArgumentParser((CleanDatasetArguments))

    clean_dataset_args = parser.parse_args()
    logger.info(clean_dataset_args)

    CleanDatasetFactory(clean_dataset_args=clean_dataset_args).up_dataset_task()

if __name__ == "__main__":
    auto_dataset()
