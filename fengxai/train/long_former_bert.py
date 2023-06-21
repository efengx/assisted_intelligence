import torch
import numpy as np
import evaluate
import logging
import sys
import os
# TODO: 后续可以考虑使用 transformers 中的方法替代掉，减少额外的依赖
import time
import graphviz

from transformers import (
    MegatronBertConfig, 
    MegatronBertModel,
    BertTokenizer,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    PreTrainedTokenizer
)
from datasets import (
    load_dataset
)
from huggingface_hub import (
    create_repo,
    HfApi,
)
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
from datetime import datetime
from fengxai.utils.log_center import create_logger

# logger = logging.getLogger(__name__)
logger = create_logger()

class ProfCallback(TrainerCallback):
    """
    推理回调
    """
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def build_dataset(
        tokenizer: PreTrainedTokenizer,
        data_args,
        model_args,
        model_max_length: int=1024,
    ):
    """
    下载并组装数据集
    """
    logger.warning("== 加载数据集 ===")
    raw_dataset=load_dataset(
        data_args.dataset_name, 
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token else None,
    )
    logger.info(raw_dataset)

    raw_dataset["train"]=raw_dataset['train'].select(range(2))
    raw_dataset["validation"]=raw_dataset['validation'].select(range(2))
    raw_dataset["test"]=raw_dataset['test'].select(range(2))

    def format_dataset(example):
        if example["target"] == "人类写的":
            example["label"] = 1
        elif example["target"] == "ai写的":
            example["label"] = 0
        else:
            logger.error("无效的target")
        logger.info(example)
        return example
    format_dataset = raw_dataset.map(
        format_dataset,
        remove_columns="target"
    )
    logger.info(raw_dataset)
    logger.info(raw_dataset["train"][:1])

    def tokenizer_fun(example):
        return tokenizer(
            example["text"],
            max_length=model_max_length,
            padding="max_length",
            truncation=True,
        )
    tokenized_dataset = format_dataset.map(tokenizer_fun, batched=True)
    logger.info(tokenized_dataset)
    logger.info(tokenized_dataset["train"][:1])

    logger.info(f"input_ids len = {len(tokenized_dataset['train'][:1]['input_ids'][0])}")
    logger.info(f"attention_mask len = {len(tokenized_dataset['train'][:1]['attention_mask'][0])}")
    
    return tokenized_dataset

def long_former_bert_train(
        model_args, data_args, training_args    
    ):
    ndt=datetime.now()

    if not os.path.exists(f"{training_args.clone_repo}/logs"):
        os.makedirs(f"{training_args.clone_repo}/logs")
    logging.basicConfig(
        # 日志配置
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{training_args.clone_repo}/logs/training-{ndt.year}{ndt.month:02d}{ndt.day}.log")
        ],
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )

    logger.info("***************** 载入模型: 分词器; 模型 *****************")
    tokenizer = BertTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token and model_args.use_auth_token else None,
    )

    # id2label在开始训练模型之前，使用和创建预期 ID 到其标签的映射label2id：
    id2label = {0: "AI write", 1: "Human write"}
    label2id = {"AI write": 0, "Human write": 1}
    """
    AutoModelForSequenceClassification 类加载 DistilBertForSequenceClassification 类作为基础模型。 
    由于 AutoModelForSequenceClassification 不接受参数“num_labels”，
    它被传递给接受它的基础类 DistilBertForSequenceClassification。
    
    to("cuda") 表示使用 gpu 加载
    """
    # LongformerZhForMaksedLM.from_pretrained('ValkyriaLenneth/longformer_zh')
    # model = MegatronBertModel.from_pretrained(
    #     model_args.model_name_or_path,
    #     # 必须的，作用：替换掉原来的模型的分类头部，根据训练的数据集重新生成新的二分类头部
    #     num_labels=2,
    #     id2label=id2label,
    #     label2id=label2id
    # )

    model=AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        # 必须的，作用：替换掉原来的模型的分类头部，根据训练的数据集重新生成新的二分类头部
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token and model_args.use_auth_token else None,
    )
    logger.info(f"model config: {model.config}")

    logger.info("************** 预处理：数据集 ********************")
    datasets = build_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        model_max_length=training_args.model_max_length
    )

    logger.info("************** 构建训练环境, 并进行训练 *****************************")
    """
    我们想在训练期间评估我们的模型。 Trainer 通过提供 compute_metrics 来支持训练期间的评估。
    评估摘要任务最常用的指标是 rogue_score，是面向召回的 Understudy for Gisting Evaluation 的缩写）。 
    该指标的行为与标准准确度不同：它将生成的摘要与一组参考摘要进行比较
    """
    # Metric
    metric=evaluate.load("accuracy")
    # 将在整个预测/标签数组的每个评估阶段结束时调用的函数，以生成指标。
    def compute_metrics(eval_pred):
        # 预测和标签被分组在一个名为 EvalPrediction 的命名元组中
        logits, labels = eval_pred
        # 获取预测分数最高的索引（即预测标签）
        predictions = np.argmax(logits, axis=1)
        # 将预测标签与参考标签进行比较
        results = metric.compute(predictions=predictions, references=labels)
        # 结果：一个包含字符串键（指标名称）和浮点数的字典
        # 值（即指标值）
        return results

    # Create Trainer instance
    # 创建 Trainer 实例
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 初始化时间，用于计算训练所用时间
    start = time.perf_counter()
    # 开启pytorch profile用于监控模型性能，通过 tensorboard 进行查看
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
        schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
        # 文件保存路径，通过 tensorboard 进行查看
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{training_args.clone_repo}/logs/resnet"),
        profile_memory=True,
        with_stack=True,
        record_shapes=True
    ) as prof:
        trainer.add_callback(ProfCallback(prof=prof))
        trainer.train()

    # 开始训练
    # trainer.train()
    # 很好，我们已经训练了我们的模型。 🎉让我们在测试集上再次评估最佳模型。
    trainer.evaluate()

    # 保存我们的分词器并创建模型卡
    logger.info(training_args.clone_repo)
    tokenizer.save_pretrained(training_args.clone_repo)
    # 存在超参数格式问题 AttributeError: 'str' object has no attribute 'value' 
    # for hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    # trainer.create_model_card()
    # 保存模型
    trainer.save_model(training_args.clone_repo)
    num=model.num_parameters()
    logger.info(num)


    logger.info("********* 读取模型结构并保存 ****************")
    # 显示模型
    encoding_input=tokenizer(
        datasets["test"][0]["text"],
        return_tensors="pt"
    )
    # 设置模型的保存格式为png
    graphviz.set_jupyter_format('png')
    # 生成模型结构
    model_graph=draw_graph(
        model, 
        input_data=encoding_input, 
        save_graph=True
    )
    visual_graph=model_graph.visual_graph
    logger.info(visual_graph)
    visual_graph.render(filename=f"{training_args.clone_repo}/logs/model_png")

    logger.info(datasets["test"][0]["text"])
    logger.info(encoding_input)
    logger.info(encoding_input.input_ids.numpy())
    logger.info(encoding_input.input_ids.numpy().shape)
    logger.info(encoding_input.input_ids.numpy().ndim)
    # 将 encoding_input 转换回文本
    tokens=tokenizer.convert_ids_to_tokens(encoding_input.input_ids.numpy()[0])
    logger.info(tokens)
    # 将文本转换成ids
    ids=tokenizer.convert_tokens_to_ids(tokens)
    logger.info(ids)

    # 该代码的本来目的是获取模型结构，但是目前无法与 huggingface 进行整合
    # tb_writer=SummaryWriter(log_dir=f"{training_args.clone_repo}/logs/rjxai_graph")
    # tb_writer.add_graph(model, encoding_input)


    logger.info("************ 评估模型 ***********************")
    # 运行推理
    # eval_result = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    # logger.info(eval_result)
    # # 将评估结果写入文件，稍后可以在 s3 输出中访问该文件
    # with open(os.path.join(f"{repository_id}/logs", "eval_results.txt"), "w") as writer:
    #     print(f"***** Eval results *****")
    #     for key, value in sorted(eval_result.items()):
    #         writer.write(f"{key} = {value}\n")
    
    # 查看测试集大小
    logger.debug(datasets["test"])
    # 使用模型进行预测
    test_prediction = trainer.predict(datasets["test"])
    # 对于每个预测，使用 argmax 创建标签（获取第一列的标签）
    test_predictions_argmax = np.argmax(test_prediction[0], axis=1)
    # 从测试集中检索参考标签
    test_references=np.array(datasets["test"]["label"])
    # 计算精度
    results = metric.compute(predictions=test_predictions_argmax, references=test_references)
    logger.info(results)


    logger.info("******* 保存模型 huggingface hub ********")
    # 保存到 huggingface
    # TODO：延迟保存会存在夸天训练无法保存的问题（该机制有待进一步完善）
    hf_host="https://huggingface.co"
    # 上传最新的模型到存储库
    logger.info("*** save model hub ***")
    api=HfApi()
    list_models=api.list_models(
        search=training_args.clone_repo, 
        token=model_args.hf_hub_token
    )
    if len(list_models) == 0:
        # 创建远程存储库（后续判断是否需要？）
        create_repo(
            training_args.clone_repo, 
            private=True,
            token=model_args.write_hf_hub_token
        )
    else:
        logger.info(list_models[0])

    # 测试使用上面的保存查看效果
    # 上传目录下的文件到存储库
    logger.info(f"folder_path:{training_args.clone_repo}")
    list_dir=os.listdir(training_args.clone_repo)
    logger.info(list_dir)
    api.upload_folder(
        folder_path=training_args.clone_repo,
        repo_id=training_args.clone_repo,
        token=model_args.write_hf_hub_token
    )
    # 查询存储库并打印(内容)
    logger.info(f"{hf_host}/{training_args.clone_repo}")

if __name__ == "__main__":
    long_former_bert_train()