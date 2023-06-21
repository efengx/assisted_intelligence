from transformers.trainer_utils import SchedulerType
from typing import Optional, List, ClassVar, Union
from dataclasses import dataclass, field
from transformers.utils import ExplicitEnum
from fengxai.utils.log_center import create_logger
from fengxai.config.base_config import BaseTrainingArguments
from transformers.trainer_utils import IntervalStrategy

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
class RlhfTrainingArguments(BaseTrainingArguments):
    
    learning_rate: Optional[float] = field(default=1e-5)
    """
    # per_device_train_batch_size：用于训练的每个 GPU/TPU 核心/CPU 的批量大小。
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    """
    # per_device_eval_batch_size：用于评估的每个 GPU/TPU 核心/CPU 的批量大小。
    """
    per_device_eval_batch_size: Optional[int] =field(default=1)
    """
    如果为 True，则使用梯度检查点以较慢的反向传递为代价来节省内存。
    """
    gradient_checkpointing: bool = field(default=True)
    """
    是否使用 fp16 16 位（混合）精度训练而不是 32 位训练。
    混合精度训练的想法是不需要所有变量都以完整（32 位）浮点精度存储。
    主要优势来自于以一半（16 位）精度保存激活。尽管梯度也是以半精度计算的，但它们会在优化​​步骤中转换回全精度，因此此处不节省内存
    """
    # fp16: bool = field(default=True)
    """
    要使用的优化器：adamw_hf、adamw_torch、adamw_torch_fused、adamw_apex_fused、adamw_anyprecision 或 adafactor。
    Adafactor 不是保留权重矩阵中每个元素的滚动平均值，而是仅存储聚合信息（滚动平均值的行和列总和），这大大减少了占用空间。Adafactor 的一个缺点是在某些情况下收敛速度可能比 Adam 慢
    """
    # default_optim = "adamw_torch"
    default_optim = 'adafactor'
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    """
    # report_to：要向其报告结果和日志的集成列表。 支持的平台是“azure_ml”、“comet_ml”、“mlflow”、“tensorboard”和“wandb”。 使用“all”报告所有已安装的集成，“none”表示没有集成。
    """
    # report_to: ClassVar[List[str]] = ["wandb", "tensorboard"]
    report_to: ClassVar[List[str]] = ["wandb"]
    """
    hub_token=model_args.hf_hub_token
    """
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "# load_best_model_at_end (default False): 训练结束时是否加载训练过程中找到的最佳模型。"}
    )
    evaluation_strategy: Optional[str] = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "评估完成（并记录）在每个eval_steps之后"}
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={"help": "记录每 500 个更新步骤之后。"}
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "保存，在每个save_steps之后。"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "如果 save_strategy='steps'，则两个检查点保存之前的更新步骤数。"}
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

    no_cuda: bool = field(
        default=True, 
        metadata={"help": "不要使用 CUDA，即使它可用，直接使用CPU"}
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
    trl-rlhf
    """
    training_step: Optional[str] = field(default="se", metadata={"help": "rlhf训练步骤，分为se, reward, rl"})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "使用lora模式"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    mixed_precision: Optional[str] = field(
        default=None,
        metadata={"help": "是否使用混合精度训练。 从“否”、“fp16”、“bf16”或“fp8”中选择。 将默认为环境变量 `ACCELERATE_MIXED_PRECISION` 中的值，这将使用当前系统的加速配置中的默认值或通过 `accelerate.launch` 命令传递的标志。 'fp16' 需要 pytorch 1.6 或更高版本。 'bf16' 需要 pytorch 1.10 或更高版本。 'fp8' 需要安装 transformers-engine。"},
    )
    tracker_project_name: Optional[str] = field(
        default="rjxai/trl", metadata={"help": "wandb project name"}
    )
    reward_model: Optional[str] = field(default="", metadata={"help": "奖励模型"})

    """
    train 自定义使用
    """
    num_proc: Optional[int] = field(default=36, metadata={"help": "核心数量，如果核可以开更多"})

    # 未使用
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to stop the PPO optimization loop early is the KL too high"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "Stop early if we exceed this value by over 50%"}
    )

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
