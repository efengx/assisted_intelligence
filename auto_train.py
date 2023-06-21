import wandb
import fengxai.config.bert_config as bert_config
import fengxai.config.base_config as base_config
import fengxai.config.llama_config as llama_config
import os
import argparse
import transformers

from fengxai.config.base_config import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser
from fengxai.utils.log_center import create_logger

# 设置环境变量：1开启离线训练
# 离线的 model 
# os.environ["TRANSFORMERS_OFFLINE"]="1"
# 离线的 datasets
# os.environ['HF_DATASETS_OFFLINE']="1"

# wandb 环境变量
os.environ['WANDB_API_KEY']="93523e57b94611e1a558a6541f834f17dd400be5"
os.environ['WANDB_PROJECT']="rjxai-default-v1"
# 仅记录最终的模型
# os.environ['WANDB_LOG_MODEL']='end'

# logger = logging.getLogger(__name__)
logger = create_logger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.wandb_project is not None:
        wandb_project=args.wandb_project
    else:
        wandb_project=os.getenv('WANDB_PROJECT')
    logger.info(f"step-notify: 告诉wandb根据环境变量信息开始{wandb_project}任务")
    wandb.init(project=wandb_project, entity="rjxai")

    logger.info("step-check: 检查pip依赖")
    import pip
    logger.info(pip.main(['list']))

    logger.info("step-check: 检查环境变量")
    logger.info(os.environ)

    parser.add_argument("--training_model", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.training_model is not None:
        logger.info(f"step-load: {args.training_model}模型参数")
        if args.training_model == "BERT" :
            parser = HfArgumentParser((ModelArguments, DataTrainingArguments, bert_config.BertTrainingArguments))
        if args.training_model == "LLAMA":
            parser = HfArgumentParser((ModelArguments, DataTrainingArguments, llama_config.LlamaTrainingArguments))
        if args.training_model == "RLHF":
            import fengxai.config.rlhf_config as rlhf_config
            parser = HfArgumentParser((ModelArguments, DataTrainingArguments, rlhf_config.RlhfTrainingArguments))
    else:
        logger.info("step-load: 默认参数")
        # 最好的方式是改变 HfArgumentParser 方法根据参数自动合并属性
        # 或者采用其他的配置文件加载方式之后合并到指定的对象中
        # 默认使用LLAMA        
        # if args.training_model == "LLAMA":
        parser = HfArgumentParser(ModelArguments, DataTrainingArguments, base_config.BaseTrainingArguments)
    
    # setup logging
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    # logger.info("step-config: 日志")
    # logging.basicConfig(
    #     datefmt='%Y-%m-%d:%H:%M:%S',
	# 	format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    logger.info(f"step-train: {training_args.training_model}模型训练")
    if training_args.training_model == 'BERT':
        from fengxai.train.single_task_train_factory import single_task_train
        single_task_train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )
    elif training_args.training_model == 'LLAMA':
        from fengxai.train.rjxai_training_factory import rjxai_train
        rjxai_train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )
    elif training_args.training_model == 'RLHF':
        from fengxai.train.trl_train_rlhf_factory import rlhf_trl_llama_train
        rlhf_trl_llama_train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )

    logger.info("step-notify: 告诉wandb完成当前任务")
    wandb.finish()

if __name__ == "__main__":
    main()
