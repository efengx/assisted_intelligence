from transformers import (
    AutoConfig,
    LlamaTokenizer,
)

from fengxai.utils.log_center import create_logger

logger = create_logger()

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def rlhf_se_tokenizer(
    training_args,
    model_args,
):  
    tokenizer_kwargs = {
        "use_fast": True,
        "revision": "main",
        "model_max_length": training_args.model_max_length,
        "use_auth_token": model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
    }
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        **tokenizer_kwargs,
    )
    logger.info(f"raw tokenizer={tokenizer}")
    
    auto_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
    )
    logger.info(f"step-init: auto config={auto_config}")

    if "LlamaForCausalLM" in auto_config.architectures[0]:
        logger.info("为 LLama 分词器设置 EOS、BOS 和 UNK 代币")
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        # required for gpt2
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"new tokenizer={tokenizer}")

    return dict(
        tokenizer=tokenizer
    )