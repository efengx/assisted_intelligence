import math

# from accelerate import Accelerator
from fengxai.dataset.build_dataset import maker_data_module, make_supervised_data_module
from peft import PeftModel, TaskType, LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    AutoConfig,
)
from torch.utils.data.dataloader import DataLoader
from fengxai.utils.print_utils import print_source
from fengxai.archive.hub import trainer_save
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import  send_example_telemetry
from transformers.utils.versions import require_version
from fengxai.utils.log_center import create_logger
from fengxai.model.build_model import adam_8bit_optim
from sklearn.metrics import accuracy_score

logger = create_logger()


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return {
        "accuracy": float(
            accuracy_score(labels, preds, normalize=True, sample_weight=None)
        )
    }

def rjxai_train(
        model_args, data_args, training_args
    ):
    # logger.info("************** 构建 accelerator ***************")
    # accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    logger.info("**************** 载入模型：载入分词器/载入模型 ********************")
    print_source()
    config_kwargs = {
        "revision": "main",
        "use_auth_token": model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
    }
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        **config_kwargs,
    )

    logger.info("通过config加载模型，通过model_args.model_name_or_path加载对应模型的权重")
    # model = AutoModelForCausalLM.from_pretrained(
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
        # # 禁用环境，降低内存消耗
        # use_cache=False,
        # load_in_8bit=True,
        # torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True,
        # device_map="auto"
    )
    logger.info(f"model.config: {model.config}")
    logger.info(f"model结构: {model}")
    # 为 int8 准备训练模型
    # model = prepare_model_for_int8_training(model)

    print_source()
    tokenizer_kwargs = {
        "use_fast": True,
        "revision": "main",
        "model_max_length": training_args.model_max_length,
        "use_auth_token": model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
    }
    # tokenizer = AutoTokenizer.from_pretrained(
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        **tokenizer_kwargs,
    )
    logger.info(f"tokenizer config: {tokenizer}")

    # with training_args.main_process_first(desc="dataset load and map tokenization"):
    logger.info("**************** 预处理数据集 ***************")
    data_module, _ = make_supervised_data_module(
    # data_module, _, train_dataloader, eval_dataloader = maker_data_module(
        tokenizer=tokenizer, 
        model=model,
        data_args=data_args, 
        model_args=model_args,
    )
    print_source()

    # adam_bnb_optim = adam_8bit_optim(
    #     model=model,
    #     training_args=training_args,
    # )

    # dataloader = DataLoader(data_module["train_dataset"], batch_size=training_args.per_device_train_batch_size)

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)
    
    # model.train()
    # for step, batch in enumerate(dataloader, start=1):
    #     loss = model(**batch).loss
    #     loss = loss / training_args.gradient_accumulation_steps
    #     accelerator.backward(loss)
    #     if step % training_args.gradient_accumulation_steps == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    
    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(
            model,
            training_args.peft_path
        )
    else:
        logger.info("Init new peft model")
        # peft_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type=TaskType.SEQ_CLS,
        # )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # 微调的目标模型
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            inference_mode=False, 
            r=8,
            lora_alpha=32., 
            lora_dropout=0.05,
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, peft_config)

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

        logger.info(f"model + peft结构: {model}")
        logger.info("验证哪些模块在lora")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name, param.shape)
        logger.info("************** 模型可训练参数量 ********************")
        model.print_trainable_parameters()
    
    # 需要使用 accelerator
    # logger.info("************* 模型在设备上的分布 *****************")
    # logger.info(model.hf_device_map)

    logger.info("************** 加载训练环境 **************")
    print_source()
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
        # optimizers=(adam_bnb_optim, None),
    )

    if training_args.do_train:
        logger.info("*************** 开始训练 ****************")
        print_source()
        train_result = trainer.train()

        # if training_args.do_save:
        #     logger.info("*********** 保存模型 **************")
        #     trainer.save_model(output_dir=training_args.output_dir)
        #     # trainer.push_to_hub(output_dir=training_args.output_dir)

        metrics = train_result.metrics
        logger.info(f"train metrics：{metrics}")
        metrics["train_samples"] = len(data_module["train_dataset"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] =len(data_module["eval_dataset"])
        try:
            perplexity = math.exp(metrics["eval_loss"])
            logger.info(f"困惑度为：{perplexity}")
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info(f"eval metrics：{metrics}")

    if training_args.do_save:
        trainer_save(
            trainer=trainer,
            training_args=training_args,
            model_args=model_args,
        )

if __name__ == "__main__":
    rjxai_train()
