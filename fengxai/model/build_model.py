# import bitsandbytes as bnb
import os
import torch
import evaluate
import pandas as pd

from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
from transformers import PreTrainedModel
from fengxai.utils.log_center import create_logger
from transformers import (
    AutoModelForSequenceClassification,
    get_scheduler,
    AdamW,
    pipeline,
    AutoModelForCausalLM,
)
from fengxai.utils.print_utils import print_source
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorWithPadding, get_linear_schedule_with_warmup
from accelerate import Accelerator
from trl import PPOConfig, SFTTrainer, set_seed
from trl.core import LengthSampler

logger = create_logger()

def _logger_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def _collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def adam_8bit_optim(
    model: PreTrainedModel,
    training_args,
):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    logger.info(decay_parameters)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    logger.info(decay_parameters)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    # adam_bnb_optim = bnb.optim.Adam8bit(
    #     optimizer_grouped_parameters,
    #     betas=(training_args.adam_beta1, training_args.adam_beta2),
    #     eps=training_args.adam_epsilon,
    #     lr=training_args.learning_rate,
    # )

    # return adam_bnb_optim

def test_train(
        model_args,
        train_dataloader,
        eval_dataloader,
    ):
    logger.info("测试train过程")
    # accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=2
    )
    logger.info(model)
    
    logger.info("加载优化器optimizer")
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    #     train_dataloader, eval_dataloader, model, optimizer
    # )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger.info(f"训练总次数：{num_training_steps}")

    progress_bar = tqdm(range(num_training_steps))
    logger.info("设置为训练模式")
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss
            loss = outputs.loss
            loss.backward()
            # accelerator.backward(loss)

            # 优化器
            optimizer.step()
            # lr调度器
            lr_scheduler.step()
            # 将所有优化的梯度设置torch.Tensor为零
            optimizer.zero_grad()
            progress_bar.update(1)
        logger.info(f"epoch：{epoch}")


def accelerator_train(
    model: PreTrainedModel,
    config: PPOConfig,
    accelerator,
    training_args,
    train_dataloader,
    eval_dataloader,
):  
    logger.info("加载evaluate")
    metric = evaluate.load("glue", "mrpc")

    num_training_steps = training_args.num_train_epochs * len(train_dataloader)
    logger.info(f"step-info: 训练总次数={num_training_steps}")

    logger.info("step-info: 加载优化器optimizer，AdamW会根据优化函数调整模型参数")
    optimizer = AdamW(params=model.parameters(), lr=config.learning_rate)

    logger.info("step-info: 加载lr调度器，调度器的主要作用是去掉噪声")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * training_args.num_train_epochs) // training_args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    num_train_epochs = int(training_args.num_train_epochs)
    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            # batch.to(accelerator.device)
            outputs = model(**batch)
            # loss
            loss = outputs.loss
            loss = loss / training_args.gradient_accumulation_steps
            # loss.backward()
            accelerator.backward(loss)
            # if step % training_args.gradient_accumulation_steps == 0:
            # 优化器
            optimizer.step()
            # lr调度器
            lr_scheduler.step()
            # 在epochs结束后：将所有优化的梯度设置torch.Tensor为零
            optimizer.zero_grad()
        
        model.eval()
        for batch in eval_dataloader:
            # batch.to(accelerator.device)
            with torch.no_grad():
                outputs=model(**batch)
            predictions=outputs.logits.argmax(dim=-1)
            predictions, references=accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # logger.info(f"epoch {epoch}:", eval_metric)

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)


def rlhf_se_train(
    training_args,
    model_args,
    tokenizer: PreTrainedTokenizer,
    train_dataset, 
    valid_dataset,
):  
    logger.info("step-train: 训练se")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if training_args.use_lora:
        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info(f"step-init: lora_config={peft_config}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
        # load_in_8bit=True,
        # device_map="auto",
    )
    logger.info(f"step-init: model={model}")
    print_source()

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        packing=True,
    )

    _logger_trainable_parameters(trainer.model)

    logger.info("step-train")
    trainer.train()

    logger.info(f"model to save: {trainer.model.modules_to_save}")

    return dict(
        trainer=trainer
    )


def rlhf_inspection(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sentiment_pipe,
    dataset,
    device,
    output_length_sampler,
    sent_kwargs, 
    gen_kwargs,
):
    logger.info("开始inspection")
    #### get a batch from the dataset
    bs = 16
    game_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref, response_tensors = [], []

    #### get response from gpt2 and gpt2_ref
    for i in range(bs):
        gen_len = output_length_sampler()
        output = ref_model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors_ref.append(output)
        output = model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors.append(output)

    #### decode responses
    game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

    #### sentiment analysis of query/response pairs before/after
    texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

    texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)
    logger.info(df_results)
    
    logger.info("mean:")
    logger.info(df_results[["rewards (before)", "rewards (after)"]].mean())
    logger.info("median:")
    logger.info(df_results[["rewards (before)", "rewards (after)"]].median())


def rlhf_reward_train(
    model_args,
    training_args,
):
    logger.info(f"step-init: 奖励模型训练")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1, 
        torch_dtype=torch.bfloat16
    )
    logger.info(f"step-init: 初始化模型={model}")
    logger.info(model.config)

    if training_args.use_lora:
        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info(f"step-init: peft config={peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info(f"step-init: 合并模型 new model={model}")
        logger.info(model.config)

    return dict(
        model=model
    )


def rl_train():
    logger.info(f"step-init: 使用奖励模型对预训练模型进行微调")
