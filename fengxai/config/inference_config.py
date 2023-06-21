from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class InferenceArguments:
    
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "tokenizer路径。"}
    )
    base_model: Optional[str] = field(
        default=None,
        metadata={"help": "基础模型路径。"}
    )
    lora_model: Optional[str] = field(
        default=None,
        metadata={"help": "peft模型路径。"}
    )
    hf_hub_token: str = field(
        default='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',
        metadata={
            "help": "远程hub的读取token"
        }
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