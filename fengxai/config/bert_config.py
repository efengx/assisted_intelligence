from transformers.trainer_utils import SchedulerType
from typing import Optional, List, ClassVar, Union
from dataclasses import dataclass, field
from transformers import (TrainingArguments)
from transformers.utils import ExplicitEnum
from transformers.trainer_utils import IntervalStrategy
from fengxai.utils.log_center import create_logger
from fengxai.config.base_config import BaseTrainingArguments

logger = create_logger()

class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"

@dataclass
class BertTrainingArguments(BaseTrainingArguments):
    # """
    # 如果output_dir设置TrainingArguments为“/opt/ml/model”，
    # Trainer 将保存所有训练工件，包括日志、检查点和模型。
    # Amazon SageMaker 将整个“/opt/ml/model”目录归档并model.tar.gz在训练作业结束时将其上传到 Amazon S3

    # # output_dir：保存模型检查点的目录。
    # # output_dir=f"{self.task}-{self.save_model}",
    # """
    # output_dir: str = field(
    #     default=None,
    #     metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    # )
    # training_model: str = field(
    #     default='BERT',
    #     metadata={"help": "训练那种类型的模型，默认为BERT。目前包含两种类型：BERT, LLAMA"}
    # )
    # clone_repo: str = field(
    #     default=None,
    #     metadata={"help": "模型保存的hub地址，通常情况下在训练之后才会保存模型"}
    # )
    """
    ####
    定义我们要用于训练的超参数 (TrainingArguments)。 
    我们正在利用 Trainer 的 Hugging Face Hub 集成，在训练期间自动将我们的检查点、日志和指标推送到存储库中。
    ####
    """
    """
    # evaluation_strategy (default "no"):
    # 可能的值是：
    # “no”：训练期间不进行评估。
    # “steps”：评估完成（并记录）每个 eval_steps。
    # “epoch”：评估在每个 epoch 结束时进行。
    """
    evaluation_strategy: Optional[str] = field(
        default='steps',
    )
    """
    # eval_steps：如果 evaluation_strategy="steps"，则两次评估之间的更新步数。 如果未设置，将默认为与 logging_steps 相同的值。
    """
    eval_steps: Optional[int] = field(
        default=50,
    )
    logging_steps: int = field(
        default=50, 
        metadata={"help": "Log every X updates steps."}
    )
    """
    # logging_strategy（默认值：“steps”）：训练期间采用的记录策略（例如用于记录训练损失）。 可能的值是：
    # “no”：训练期间不进行日志记录。
    # “epoch”：记录在每个纪元结束时完成。
    # “steps”：每 logging_steps 记录一次。
    """
    logging_strategy: Optional[str] = field(
        default='steps'
    )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "日志保存路径"}
    )
    
    # metadata={
    #     "help": (
    #         "save_strategy（默认“steps”）：训练期间采用的检查点保存策略。 可能的值是："
    #         "no: 训练期间不保存。"
    #         "epoch: 保存在每个 epoch 结束时完成。"
    #         "steps: 每 save_steps 保存一次（默认 500）"
    #     )
    # },
    save_strategy: Union[IntervalStrategy, str] = field(
        default=IntervalStrategy.STEPS
    )
    # metadata={
    #     "help", "save_steps（默认值：500）：如果 save_strategy='steps'，则两次检查点保存之前的更新步骤数。"
    # }
    # metadata 存在错误，需要研究下 metadata 机制
    save_steps: int = field(
        default=100,
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "保存3个 checkpoint"}
    )


    """
    将在训练开始时设置的随机种子。 为确保跨运行的可重复性，如果模型具有一些随机初始化的参数，请使用 [`~Trainer.model_init`] 函数实例化模型。
    """
    # seed: int = field(
    #     default=7, 
    #     metadata={"help": "Random seed that will be set at the beginning of training."}
    # )
    """
    # learning_rate（默认 5e-5）：AdamW 优化器的初始学习率。 Adam 算法与权重衰减修正在论文解耦权重衰减正则化中介绍。
    之前训练用的是：2e-5
    """
    learning_rate: Optional[float] = field(default=2e-5)
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        # metadata={"help": "per_device_train_batch_size：用于训练的每个 GPU/TPU 核心/CPU 的批量大小。"}
    )
    per_device_eval_batch_size: Optional[int] =field(
        default=16,
        # metadata={"help": "per_device_eval_batch_size：用于评估的每个 GPU/TPU 核心/CPU 的批量大小。"}
    )
    """
    如果为 True，则使用梯度检查点以较慢的反向传递为代价来节省内存。
    """
    # gradient_checkpointing: bool = field(default=True)
    """
    要使用的优化器：adamw_hf、adamw_torch、adamw_torch_fused、adamw_apex_fused、adamw_anyprecision 或 adafactor。
    Adafactor 不是保留权重矩阵中每个元素的滚动平均值，而是仅存储聚合信息（滚动平均值的行和列总和），这大大减少了占用空间。Adafactor 的一个缺点是在某些情况下收敛速度可能比 Adam 慢
    """
    # default_optim = "adamw_torch"
    # optim: Union[OptimizerNames, str] = field(
    #     default=default_optim,
    #     metadata={"help": "The optimizer to use."},
    # )
    num_train_epochs: Optional[float] = field(
        default=1.0,
        metadata={"help": "# num_train_epochs (default 3.0): 要执行的训练 epoch 总数（如果不是整数，将执行停止训练前最后一个 epoch 的小数部分百分比）。"}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "# load_best_model_at_end (default False): 训练结束时是否加载训练过程中找到的最佳模型。"}
    )
    metric_for_best_model: Optional[str] = field(
        default="accuracy",
        metadata={"help": "# metric_for_best_model：与load_best_model_at_end结合使用，指定用于比较两个不同模型的指标。 必须是评估返回的指标名称，带有或不带有前缀“eval_”。"}
    )
    """
    # 在主进程上使用的记录器日志级别。 可能的选择是日志级别作为字符串：'debug'、'info'、'warning'、'error' 和 'critical'，以及不设置任何内容并保持 Transformer 的当前日志级别的 'passive' 级别 库（默认情况下为“警告”）。
    # 出现错误：SyntaxError: positional argument follows keyword argument
    # log_level: "debug",
    # report_to：要向其报告结果和日志的集成列表。 支持的平台是“azure_ml”、“comet_ml”、“mlflow”、“tensorboard”和“wandb”。 使用“all”报告所有已安装的集成，“none”表示没有集成。
    """
    report_to: ClassVar[List[str]] = ["wandb", "tensorboard"]
    
    """
    值必须改为枚举类型，否则会value无法获取
    （会在 trainer.create_model_card() 中出现错误：AttributeError: 'str' object has no attribute 'value'）
    default="linear"
    """
    # lr_scheduler_type: Union[SchedulerType, str] = field(
    #     default=SchedulerType.LINEAR,
    #     metadata={"help": "The scheduler type to use."},
    # )
    """"
    # 覆盖输出目录（本次存储时使用）
    # overwrite_output_dir=True
    # 在训练完成后将模型写入存储库,训练期间推送到hub    
    """
    push_to_hub: Optional[bool] = field(default=False)
    hub_strategy: Optional[str] = field(default="every_save")
    hub_model_id: Optional[str] = field(default=None)


    """
    hub_token=model_args.hf_hub_token
    """
    """
    开启训练流程
    """
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "hf不会使用该参数，该参数在 ml 环节中使用"}
    )
    do_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "开启PEFT；hf不会使用该参数，该参数在 ml 环节中使用。"}
    )
    """
    开启test预测流程
    """
    do_predict: Optional[bool] = field(
        default=True,
        metadata={"help": "hf不会使用该参数，该参数在 ml 环节中使用"}
    )
    """
    开启 accelerator 流程
    """
    do_accelerator: Optional[bool] = field(
        default=False,
        metadata={"help": "hf 不会使用该参数，在 ml 环节使用"}
    )
    """
    开启 tensorboard 可视化
    """
    do_visualization: Optional[bool] = field(
        default=True,
        metadata={"help": "hf 不会使用该参数，在 ml 环节使用"}
    )
    metric_type: Optional[str] = field(
        default="accuracy",
        # default="seqeval",
        metadata={"help": "指标类型；hf 不会使用该参数，在 ml 环节使用。"}
    )


    """
    需要 torch 2.0.* 以上版本支持，与 deepspeed 存在不兼容的问题，同时 sagemaker 的 torch 版本比较旧，无法兼容，暂时不使用此特征
    intel_extension_for_pytorch。
    20230521torch2.0.*存在版本冲突问题
    """
    # use_ipex: bool = field(
    #     default=True,
    #     metadata={
    #         "help": (
    #             "Use Intel extension for PyTorch when it is available, installation:"
    #             " 'https://github.com/intel/intel-extension-for-pytorch'"
    #         )
    #     },
    # )

    # fp16: bool = field(
    #   default=True,
    #   metadata={
    #       "help": (
    #           "适用于GPU，与bf16/no_cuda参数冲突，二选一： 是否使用 fp16 16 位（混合）精度训练而不是 32 位训练。"
    #           "混合精度训练的想法是不需要所有变量都以完整（32 位）浮点精度存储。"
    #           "主要优势来自于以一半（16 位）精度保存激活。尽管梯度也是以半精度计算的，但它们会在优化​​步骤中转换回全精度，因此此处不节省内存"
    #       )
    #   }
    # )

    # bf16: bool = field(
    #     default=True,
    #     metadata={
    #         "help": (
    #             "试用与CPU，与fp16参数冲突两者2选一"
    #             "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
    #             " architecture or using CPU (no_cuda). This is an experimental API and it may change."
    #         )
    #     },
    # )

    # no_cuda: bool = field(
    #     default=True, 
    #     metadata={
    #         "help": (
    #             "不要使用 CUDA，即使它可用"
    #         )
    #     }
    # )
    
    def __post_init__(self):
        if self.clone_repo is not None:
            self.logging_dir = f"{self.clone_repo}/logs"
            self.hub_model_id = self.clone_repo

        # if self.report_to is None:
        #     logger.info(
        #         "The default value for the training argument `--report_to` will change in v5 (from all installed "
        #         "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
        #         "now. You should start updating your code and make this info disappear :-)."
        #     )
        #     self.report_to = "all"
        # elif self.report_to == "none" or self.report_to == ["none"]:
        #     self.report_to = []
        # elif not isinstance(self.report_to, list):
        #     self.report_to = [self.report_to]
    