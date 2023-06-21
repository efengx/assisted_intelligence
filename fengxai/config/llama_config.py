from transformers.trainer_utils import SchedulerType
from typing import Optional, List, ClassVar, Union
from dataclasses import dataclass, field
from transformers import (TrainingArguments)
from transformers.utils import ExplicitEnum
from fengxai.utils.log_center import create_logger
from fengxai.config.base_config import BaseTrainingArguments

# logger = logging.getLogger(__name__)
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
class LlamaTrainingArguments(BaseTrainingArguments):
    """
    # per_device_train_batch_size：用于训练的每个 GPU/TPU 核心/CPU 的批量大小。
    """
    per_device_train_batch_size: Optional[int] = field(default=2)
    """
    # per_device_eval_batch_size：用于评估的每个 GPU/TPU 核心/CPU 的批量大小。
    """
    per_device_eval_batch_size: Optional[int] =field(default=2)
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "如果为 True，则使用梯度检查点以较慢的反向传递为代价来节省内存。"
        }
    )
    
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
    要使用的优化器：adamw_hf、adamw_torch、adamw_torch_fused、adamw_apex_fused、adamw_anyprecision 或 adafactor。
    Adafactor 不是保留权重矩阵中每个元素的滚动平均值，而是仅存储聚合信息（滚动平均值的行和列总和），这大大减少了占用空间。Adafactor 的一个缺点是在某些情况下收敛速度可能比 Adam 慢
    """
    # default_optim = "adamw_torch"
    # # default_optim = 'adafactor'
    # optim: Union[OptimizerNames, str] = field(
    #     default=default_optim,
    #     metadata={"help": "The optimizer to use."},
    # )
    report_to: ClassVar[List[str]] = field(
        default=["wandb", "tensorboard"],
        metadata={
            "help": "report_to：要向其报告结果和日志的集成列表。 支持的平台是“azure_ml”、“comet_ml”、“mlflow”、“tensorboard”和“wandb”。 使用“all”报告所有已安装的集成，“none”表示没有集成。"
        }
    )
    """
    hub_token=model_args.hf_hub_token
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


    """
    开启训练流程
    """
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "hf不会使用该参数，该参数在 ml 环节中使用"}
    )
    """
    开启test预测流程
    """
    do_predict: Optional[bool] = field(
        default=True,
        metadata={"help": "hf不会使用该参数，该参数在 ml 环节中使用"}
    )
    do_eval: Optional[bool] = field(
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
        default=False,
        metadata={"help": "hf 不会使用该参数，在 ml 环节使用"}
    )


    """
    是否使用 fp16 16 位（混合）精度训练而不是 32 位训练。
    混合精度训练的想法是不需要所有变量都以完整（32 位）浮点精度存储。
    主要优势来自于以一半（16 位）精度保存激活。尽管梯度也是以半精度计算的，但它们会在优化​​步骤中转换回全精度，因此此处不节省内存
    """
    # fp16: bool = field(default=True)

    no_cuda: bool = field(
        default=True, 
        metadata={"help": "不要使用 CUDA，即使它可用"}
    )
    # use_ipex: bool = field(
    #     default=True,
    #     metadata={
    #         "help": (
    #             "Use Intel extension for PyTorch when it is available, installation:"
    #             " 'https://github.com/intel/intel-extension-for-pytorch'"
    #         )
    #     },
    # )
    # bf16: bool = field(
    #     default=True,
    #     metadata={
    #         "help": (
    #             "是否使用 bf16（混合）精度而不是 32 位。 需要Ampere或更高的 NVIDIA"
    #             "体系结构或使用 CPU (no_cuda)。 这是一个实验性 API，可能会发生变化。"
    #         )
    #     },
    # )
    """
    PEFT就是采用LoRA的方式(如果存在，则使用已经微调过的LoRA)
    LoRA的权重
    """
    peft_path : Optional[str] = field(
        default=None
    )
    
    def __post_init__(self):
        if self.clone_repo is not None:
            self.logging_dir = f"{self.clone_repo}/logs"
            self.hub_model_id = self.clone_repo
            self.output_dir = self.clone_repo
        
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