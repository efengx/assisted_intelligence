import numpy as np
import evaluate
import os
import torch
# TODO: 后续可以考虑使用 transformers 中的方法替代掉，减少额外的依赖
import graphviz
# import bitsandbytes as bnb
import transformers
import datasets

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType, PromptEncoderConfig
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
from fengxai.utils.print_utils import print_summary, print_source
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from huggingface_hub import (
    login,
    HfApi,
    create_repo,
)
from torch.utils.tensorboard import SummaryWriter
# from torchview import draw_graph
# from datetime import datetime
from fengxai.utils.log_center import create_logger
from fengxai.train.prof_callback import ProfCallback
from fengxai.dataset.build_dataset import maker_data_module

logger = create_logger()

# 设置transformers的日志
transformers.utils.logging.set_verbosity_debug()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
transformers.tokenization_utils.logging.set_verbosity_debug()
datasets.utils.logging.set_verbosity_debug()

def single_task_train(
        model_args, data_args, training_args
    ):

    logger.info("***************** 载入模型: 分词器; 模型 *****************")
    print_source()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token and model_args.use_auth_token else None,
    )
    logger.info(f"tokenizer={tokenizer}")
    
    """
    AutoModelForSequenceClassification 类加载 DistilBertForSequenceClassification 类作为基础模型。 
    由于 AutoModelForSequenceClassification 不接受参数“num_labels”，
    它被传递给接受它的基础类 DistilBertForSequenceClassification。
    
    to("cuda") 表示使用 gpu 加载
    """
    if model_args.task_type == 'text-generation':
        # sentiment-analysis
        # id2label在开始训练模型之前，使用和创建预期 ID 到其标签的映射label2id：
        id2label = {0: "ai写的", 1: "人类写的"}
        label2id = {"ai写的": 0, "人类写的": 1}

        model=AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            # 必须的，作用：替换掉原来的模型的分类头部，根据训练的数据集重新生成新的二分类头部
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token and model_args.use_auth_token else None,
        )
    elif model_args.task_type == 'text2text':
        model=AutoModelForCausalLM.from_pretrained_model(
            model_args.model_name_or_path,
            use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token and model_args.use_auth_token else None,
        )

    if training_args.do_peft:
        logger.info("********* 使用 peft 训练 ***********")
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS", 
            num_virtual_tokens=20, 
            encoder_hidden_size=128 
        )
        model=get_peft_model(model, peft_config)

        logger.info("********** 模型可训练参数量 *************")
        # AttributeError: 'LongformerForSequenceClassification' object has no attribute 'print_trainable_parameters'
        model.print_trainable_parameters()

    num=model.num_parameters()
    logger.info(f"model 参数量：{num}")
    logger.info(f"model config: {model.config}")

    # accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    # with accelerator.main_process_first():
    
    if training_args.do_load_dataset:
        logger.info("************** 预处理：数据集 ********************")
        print_source()
        data_module, test_dataset, _, _ = maker_data_module(
            tokenizer=tokenizer,
            model=model,
            data_args=data_args,
            model_args=model_args
        )

    # logger.info("*************** 8-bit Adam 降低内存消耗 **************")
    # """
    # 8 位 Adam 不像 Adafactor 那样聚合优化器状态，​​而是保留完整状态并对其进行量化。
    # 量化意味着它以较低的精度存储状态，并且仅为优化而对其进行反量化。这类似于 FP16 训练背后的想法，即使用精度较低的变量可以节省内存。
    # https://huggingface.co/docs/transformers/v4.18.0/en/performance
    # 使用CPU训练不需要
    # """
    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # logger.info(decay_parameters)
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # logger.info(decay_parameters)
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
    #         "weight_decay": training_args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer_kwargs = {
    #     "betas": (training_args.adam_beta1, training_args.adam_beta2),
    #     "eps": training_args.adam_epsilon,
    # }
    # optimizer_kwargs["lr"] = training_args.learning_rate
    # adam_bnb_optim = bnb.optim.Adam8bit(
    #     optimizer_grouped_parameters,
    #     betas=(training_args.adam_beta1, training_args.adam_beta2),
    #     eps=training_args.adam_epsilon,
    #     lr=training_args.learning_rate,
    # )

    if training_args.do_train:
        print_source()
        if training_args.do_accelerator:
            logger.info("************** 使用 accelertaor 构建训练环境，并进行训练 **************")
            # dataloader = DataLoader(datasets, batch_size=training_args.per_device_train_batch_size)

            # if training_args.gradient_checkpointing:
            #     model.gradient_checkpointing_enable()
            
            # accelerator = Accelerator(fp16=training_args.fp16)
            # model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

            # model.train()
            # for step, batch in enumerate(dataloader, start=1):
            #     loss = model(**batch).loss
            #     # gradient_accumulation_steps: 在执行向后/更新传递之前累积梯度的更新步骤数。
            #     loss = loss / training_args.gradient_accumulation_steps
            #     accelerator.backward(loss)
            #     if step % training_args.gradient_accumulation_steps == 0:
            #         optimizer.step()
            #         optimizer.zero_grad()
            
            # # 使用wait_for_everyone()确保所有进程在继续之前加入该点。
            # accelerator.wait_for_everyone()
            # # 在保存之前使用unwrap_model()来删除在分布式过程中添加的所有特殊模型包装器。
            # unwrapped_model = accelerator.unwrap_model(model)
            # state_dict = unwrapped_model.state_dict()
            # logger.info(state_dict)
            # # 保存模型（只保存一次）
            # accelerator.save(state_dict.state_dict(), training_args.clone_repo)

        else:
            """
            我们想在训练期间评估我们的模型。 Trainer 通过提供 compute_metrics 来支持训练期间的评估。
            评估摘要任务最常用的指标是 rogue_score，是面向召回的 Understudy for Gisting Evaluation 的缩写）。 
            该指标的行为与标准准确度不同：它将生成的摘要与一组参考摘要进行比较
            """
            logger.info("************** 使用 Trainer 构建训练环境, 并进行训练 ******************")
            if training_args.metric_type == "seqeval":
                logger.info("加载seqeval包含多个指标（精度、准确性、F1 和召回率）的框架，用于评估序列标记任务。")
                metric = evaluate.load(training_args.metric_type)
                label_list = [
                    "ai写的",
                    "人类写的",
                ]
                # 评估 roc_auc
                roc_auc_score = evaluate.load("roc_auc")
                refs=[1, 0, 1, 1, 0, 0]
                pred_scores=[0.5, 0.2, 0.99, 0.3, 0.1, 0.7]
                results = roc_auc_score.compute(references=refs, prediction_scores=pred_scores)
                logger.info(round(results['roc_auc'], 2))

                def compute_metrics(pred):
                    logger.debug(f"pred:\n{pred}")
                    labels = pred.label_ids
                    preds = pred.predictions.argmax(-1)
                    logger.debug(f"new labels: {labels}")
                    logger.debug(f"new preds: {preds}")
                    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
                    acc = accuracy_score(labels, preds)
                    return {
                        'accuracy': acc,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    }
            else:
                logger.info("加载accuracy")
                # Metric
                metric=evaluate.load("accuracy", "f1")
                # metric = evaluate.load("accuracy", "f1", "glue", "mrpc")
                # 将在整个预测/标签数组的每个评估阶段结束时调用的函数，以生成指标。
                def compute_metrics(eval_pred):
                    # 预测和标签被分组在一个名为 EvalPrediction 的命名元组中
                    logits, labels = eval_pred
                    # 获取预测分数最高的索引（即预测标签）
                    predictions = np.argmax(logits, axis=-1)
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
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                **data_module
                # 开启 int8 adam
                # optimizers=(adam_bnb_optim, None),
            )

            print_source()
            if training_args.do_visualization:
                logger.info("加载tensorboard训练过程可视化，会影响性能，生成环境建议关闭")
                # 初始化时间，用于计算训练所用时间
                # start = time.perf_counter()
                # 开启pytorch profile用于监控模型性能，通过 tensorboard 进行查看
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU, 
                        torch.profiler.ProfilerActivity.CUDA
                    ], 
                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
                    # 文件保存路径，通过 tensorboard 进行查看
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{training_args.clone_repo}/logs/resnet"),
                    profile_memory=True,
                    with_stack=True,
                    record_shapes=True
                ) as prof:
                    # 添加训练的回调
                    trainer.add_callback(ProfCallback(prof=prof))
                    train_result = trainer.train()

                logger.info("********* 读取模型结构并保存 ****************")
                encoding_input=tokenizer(
                    test_dataset[0][data_args.index_input],
                    return_tensors="pt"
                )
                logger.info(f"encoding_input: {encoding_input}")
                # 该代码的本来目的是获取模型结构，但是目前无法与 huggingface 进行整合
                # tb_writer=SummaryWriter(log_dir=f"{training_args.clone_repo}/logs/rjxai_graph")
                # tb_writer.add_image("train_input", encoding_input)
                # 错误：RuntimeError: Type 'Tuple[str, str]' cannot be traced. Only Tensors and  (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced
                # tb_writer.add_graph(model, encoding_input)

                logger.info("目前无法在sagemaker上apt-get graphviz导致无法使用该库，需要寻找另外的方案")
                # 设置模型的保存格式为png
                # graphviz.set_jupyter_format('png')
                # 生成模型结构
                # model_graph=draw_graph(
                #     model, 
                #     input_data=encoding_input, 
                #     save_graph=True
                # )
                # visual_graph=model_graph.visual_graph
                # logger.info(visual_graph)
                # visual_graph.render(filename=f"{training_args.clone_repo}/logs/model_png")

                logger.info(f"原始数据集：{test_dataset[0][data_args.index_input]}")
                logger.info(f"encoding input: {encoding_input}")
                logger.info(encoding_input.input_ids.numpy())
                logger.info(encoding_input.input_ids.numpy().shape)
                logger.info(encoding_input.input_ids.numpy().ndim)
                
                tokens=tokenizer.convert_ids_to_tokens(encoding_input.input_ids.numpy()[0])
                logger.info(f"将 ids 转换成 tokens 文本：ids_to_tokens = {tokens}")
                ids=tokenizer.convert_tokens_to_ids(tokens)
                logger.info(f"将tokens转换成ids：tokens_to_ids = {ids}")
            else:
                logger.info("使用快速训练")
                train_result = trainer.train()

            logger.info(f"训练结果：{train_result}")
            logger.info("******** 显示内存使用情况 **********")
            print_summary(train_result)

            logger.info("******** 运行评估循环并返回指标 **********")
            # 很好，我们已经训练了我们的模型。 🎉让我们在测试集上再次评估最佳模型。
            eval_result = trainer.evaluate()
            logger.info(eval_result)

    if training_args.do_predict:
        logger.info("************ 返回测试集上的预测（如果标签可用，则带有度量） ********************")
        # 运行推理
        # eval_result = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        # logger.info(eval_result)
        # # 将评估结果写入文件，稍后可以在 s3 输出中访问该文件
        # with open(os.path.join(f"{repository_id}/logs", "eval_results.txt"), "w") as writer:
        #     print(f"***** Eval results *****")
        #     for key, value in sorted(eval_result.items()):
        #         writer.write(f"{key} = {value}\n")
        
        # 查看测试集大小
        logger.info(test_dataset)
        # 使用模型进行预测
        test_prediction = trainer.predict(test_dataset)
        # 对于每个预测，使用 argmax 创建标签（获取第一列的标签
        logger.info(test_prediction)
        
        logger.info(test_prediction.predictions.shape)
        logger.info(test_prediction.label_ids.shape)
        logger.info(test_prediction[0])
        
        test_predictions_argmax = np.argmax(test_prediction[0], axis=1)
        # 从测试集中检索参考标签
        test_references=np.array(test_dataset["labels"])
        logger.info(test_predictions_argmax)
        logger.info(test_references)
        # 计算精度
        results = metric.compute(predictions=test_predictions_argmax, references=test_references)
        logger.info(f"test 精度值：{results}")


    # model.save_pretrained(training_args.clone_repo, push_to_hub=True)
    model.push_to_hub(training_args.clone_repo, use_auth_token=model_args.write_hf_hub_token)
    tokenizer.push_to_hub(training_args.clone_repo, use_auth_token=model_args.write_hf_hub_token)

    if training_args.do_save:
        # 保存我们的分词器并创建模型卡
        logger.info(training_args.clone_repo)
        trainer.save_model(training_args.clone_repo)

        # login(token=model_args.write_hf_hub_token, add_to_git_credential=True)
        # trainer.push_to_hub(training_args.clone_repo)
        with training_args.main_process_first(desc="save training model"):
            # with accelerator.main_process_first():
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
    single_task_train()