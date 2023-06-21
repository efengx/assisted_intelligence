from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class CleanDatasetArguments:

    input_path: Optional[str] = field(
        default="/Users/ofengx/Desktop/ai/fengx_dataset/input/text-classification",
        metadata={
            "help": (
                "输入的路径"
            )
        },
    )
    file_name_list: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "合并的文件列表"
            )
        },
    )
    input_title_list: Optional[List[str]] = field(
        default_factory=lambda: ["text"],
        metadata={
            "help": (
                "input的标题列表，旧的会覆盖新的"
            )
        },
    )
    output_title_list: Optional[List[str]] = field(
        default_factory=lambda: ["labels"],
        metadata={
            "help": (
                "input的标题列表，旧的会覆盖新的"
            )
        },
    )
    clone_repo: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型保存的hub地址，通常情况下在训练之后才会保存模型"
            )
        },
    )
    do_save: Optional[bool] = field(
        default=False,
        metadata={"help": "开启PEFT；hf不会使用该参数，该参数在 ml 环节中使用。"}
    )
    task_name: Optional[str] = field(
        # default="TO_CLEAN",
        default="MERGE_SPLIT_SAVE",
        metadata={
            "help": (
                "任务名称, 可能的值为 TO_CLEAN, TEST"
            )
        },
    )
    write_hf_hub_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "远程hub的写入token"
        }
    )
    is_save_hf: Optional[bool] = field(
        default=True,
        metadata={"help": "是否保存到hf hub。"}
    )
    train_dataset_lenght: Optional[int] = field(
        default=None,
        metadata={"help": "训练集的长度。"}
    )
    input_max_length: Optional[int] = field(
        default=512,
        metadata={"help": "input的最大长度"}
    )