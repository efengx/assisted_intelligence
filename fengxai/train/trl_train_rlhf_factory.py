import torch
import os

from accelerate import Accelerator
from transformers import (
    pipeline, 
    AutoTokenizer, 
    Adafactor, 
    set_seed, 
    get_linear_schedule_with_warmup,
    LlamaTokenizer,
    AutoConfig,
)
from torch.optim import AdamW
from tqdm import tqdm
from trl import (
    PPOTrainer, 
    PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from fengxai.archive.hub import rlhf_se_save
from peft import LoraConfig
from fengxai.dataset.build_dataset import make_supervised_data_module, rlhf_se_dataset, rlhf_reward_dataset
from fengxai.tokenizer.build_tokenizer import rlhf_se_tokenizer
from trl.core import (
    LengthSampler
)
from fengxai.model.build_model import accelerator_train, rlhf_se_train, rlhf_inspection
from huggingface_hub import (
    login,
    create_repo,
    HfApi,
)
from fengxai.utils.log_center import create_logger

logger = create_logger()

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def _rlhf_se_train(
    model_args, data_args, training_args
):
    logger.info("step(rlhf-se): 初步预训练")
    dict_tokenizer=rlhf_se_tokenizer(
        model_args=model_args,
        training_args=training_args,
    )
    tokenizer=dict_tokenizer['tokenizer']
    logger.info(f"step-init(tokenizer): 生成的tokenizer为={tokenizer}")

    dataset=rlhf_se_dataset(
        tokenizer=tokenizer,
        data_args=data_args, 
        model_args=model_args,
    )
    logger.info(f"step-init(dataset): 生成的数据为--")
    logger.info(dataset["train_dataset"].dataset)
    logger.info(dataset["valid_dataset"].dataset)

    trainer_dict=rlhf_se_train(
        model_args=model_args,
        training_args=training_args,
        tokenizer=tokenizer,
        **dataset,
    )

    rlhf_se_save(
        trainer=trainer_dict["trainer"],
        training_args=training_args,
        model_args=model_args,
    )

def _rlhf_reward_train(
    model_args, data_args, training_args
):
    logger.info("step(rlhf-reward): 训练奖励模型")
    dict_tokenizer=rlhf_se_tokenizer(
        model_args=model_args,
        training_args=training_args,
    )
    tokenizer=dict_tokenizer['tokenizer']
    logger.info(f"step-init(tokenizer): 生成的tokenizer为={tokenizer}")

    dataset=rlhf_reward_dataset(
        tokenizer=tokenizer,
        data_args=data_args, 
        model_args=model_args,
    )
    logger.info(f"step-init(dataset): 生成的数据为--")
    logger.info(dataset["train_dataset"].dataset)
    logger.info(dataset["valid_dataset"].dataset)


def _rlhf_rl_train(
    model_args, data_args, training_args
):
    logger.info("step(rlhf-rl): 强化学习方式训练模型")



def rlhf_trl_llama_train(
    model_args, data_args, training_args
):  
    if training_args.training_step == "se":
        _rlhf_se_train(model_args, data_args, training_args)
    elif training_args.training_step == "reward":
        _rlhf_reward_train(model_args, data_args, training_args)
    elif training_args.training_step == "rl":
        _rlhf_rl_train(model_args, data_args, training_args)

if __name__ == "__main__":
    rlhf_trl_llama_train()
