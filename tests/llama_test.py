import unittest
import os

from fengxai.config.base_config import ModelArguments, DataTrainingArguments, FxTrainingArguments
from fengxai.dataset.build_dataset import make_supervised_data_module, maker_data_module
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers import HfArgumentParser
from fengxai.utils.log_center import create_logger

logger = create_logger()

# 设置环境变量：1开启离线训练
# 离线的 model 
os.environ["TRANSFORMERS_OFFLINE"]="1"
# 离线的 datasets
os.environ['HF_DATASETS_OFFLINE']="1"

class LlamaTestCase(unittest.TestCase):

    def setUp(self) -> None:
        model_args = ModelArguments()
        data_args = DataTrainingArguments()
        training_args = FxTrainingArguments()

        model_args.use_auth_token = True
        # model_args.model_name_or_path = "rjx/rjxai-zh-llama-7b-v1"
        model_args.model_name_or_path = "severinsimmler/xlm-roberta-longformer-base-16384"
        data_args.dataset_name = "rjx/ai-and-human-20"
        data_args.index_input = "text"
        data_args.index_output = "labels"
        data_args.max_train_samples = 5
        data_args.max_eval_samples = 2
        data_args.max_test_samples = 2

        self.data_args = data_args
        self.model_args = model_args
        logger.info(self.data_args)
        logger.info(self.model_args)

        logger.info("加载tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
            use_fast=True,
        )
        logger.info(f"tokenizer: {self.tokenizer}")

        logger.info("加载model config")
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
        )
        logger.info(f"model config: {config}")
        
        # logger.info("加载model")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
        #     # 禁用环境，降低内存消耗
        #     use_cache=False,
        #     # load_in_8bit=True,
        #     # torch_dtype=torch_dtype,
        #     # low_cpu_mem_usage=True,
        #     # device_map="auto"
        # )
        # logger.info(f"model: {self.model}")

        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_tokenizer_dataset(self):
        logger.info("测试train过程")
        logger.info("测试tokeninzer dataset")
        data_module, _ = make_supervised_data_module(
            tokenizer=self.tokenizer, 
            model=None,
            data_args=self.data_args, 
            model_args=self.model_args,
        )

        logger.info(data_module)

if __name__ == '__main__':
    unittest.main()