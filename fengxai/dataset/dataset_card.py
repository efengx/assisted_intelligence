from typing import Optional, field
from dataclasses import dataclass

@dataclass
class DatasetCard:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "数据集名称"
            )
        },
    )
