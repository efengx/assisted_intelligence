import logging

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments
)
from transformers.utils.versions import require_version
from fengxai.utils.log_center import create_logger

logger = create_logger()

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default="BERT",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    hf_hub_token: str = field(
        default=None,
        metadata={
            "help": "远程hub的读取token"
        }
    )
    write_hf_hub_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "远程hub的写入token"
        }
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "设置模型的类型，不同的类型会有不同的模型载入机制"
        }
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="rjx/ai-and-human-20", 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    # preprocessing_num_workers: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "The number of processes to use for the preprocessing."},
    # )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    index_input: str = field(default='text', metadata={"help": "数据集表示输入的列名"})

    index_output: str = field(default='labels', metadata={"help": "数据集表示输出的列名"})
    
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class BaseTrainingArguments(TrainingArguments):
    """
    如果output_dir设置TrainingArguments为“/opt/ml/model”，
    Trainer 将保存所有训练工件，包括日志、检查点和模型。
    Amazon SageMaker 将整个“/opt/ml/model”目录归档并model.tar.gz在训练作业结束时将其上传到 Amazon S3
    # 减少 aws 的依赖，不需要上传到 s3, 但是可以考虑将 s3 作为cache目录（后续扩展考虑）
    """
    output_dir: str = field(
        # default="/opt/ml/model",
        default="ml/model",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default=None,
        metadata={"help": "Tensorboard log dir."},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    training_model: str = field(
        default='BERT',
        metadata={"help": "训练那种类型的模型，默认为BERT。目前包含两种类型：BERT, LLAMA"}
    )
    clone_repo: str = field(
        default=None,
        metadata={"help": "模型保存的hub地址，通常情况下在训练之后才会保存模型"}
    )


    do_load_dataset: bool = field(
        default=True,
        metadata={"help": "开启数据集加载；hf不会使用该参数，该参数在 ml 环节中使用。"}
    )
    """
    开启训练流程
    """
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "开启训练流程；hf不会使用该参数，该参数在 ml 环节中使用。"}
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
        metadata={"help": "hf不会使用该参数，该参数在 ml 环节中使用。"}
    )
    """
    开启 tensorboard 可视化
    """
    do_visualization: Optional[bool] = field(
        default=False,
        metadata={"help": "hf 不会使用该参数，在 ml 环节使用"}
    )
    do_save: Optional[bool] = field(
        default=True,
        metadata={"help": "hf 不会使用该参数，在 ml 环节使用"}
    )
    """
    开启 accelerator 流程
    """
    do_accelerator: Optional[bool] = field(
        default=False,
        metadata={"help": "开启分布式训练；hf 不会使用该参数，在 ml 环节使用。"}
    )

    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "wandb项目名称"}
    )


