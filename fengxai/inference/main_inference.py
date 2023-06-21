import fengxai.inference.code.llama_inference as llama_inference
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import  PeftModel
from fengxai.utils.log_center import create_logger
from huggingface_hub import (
    login,
)

logger = create_logger()

def main_inference(
    inference_args,
):
    logger.info("step-init: 开始推理")
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    # model, tokenizer=llama_inference.model_fn(
    #     model_dir=inference_args.model_dir,
    # )

    tokenizer = LlamaTokenizer.from_pretrained(
        inference_args.tokenizer_path,
        use_auth_token=inference_args.hf_hub_token if inference_args.use_auth_token and inference_args.hf_hub_token is not None else None,
    )

    base_model = LlamaForCausalLM.from_pretrained(
        inference_args.base_model, 
        use_auth_token=inference_args.hf_hub_token if inference_args.use_auth_token and inference_args.hf_hub_token is not None else None,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    logger.info(f"Vocab of the base model: {model_vocab_size}")
    logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        logger.info("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if inference_args.lora_model is not None:
        logger.info("loading peft model")
        login(token=inference_args.hf_hub_token, add_to_git_credential=True)
        model = PeftModel.from_pretrained(
            base_model,
            inference_args.lora_model,
            use_auth_token=inference_args.hf_hub_token if inference_args.use_auth_token and inference_args.hf_hub_token is not None else None,
            # torch_dtype=load_type,
            # device_map='auto',
        )
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()

    logger.info("step-init: 加载数据")
    example={'text': ['惊艳亮相 索尼爱立信W595全黑版曝光\u3000\u3000虽然决定人们购买欲望的最大因素是手机功能及价格，但有时候不同寻常的色彩款式同样也能吸引众多眼球的关注，尤其是对一些正在热销的手机而言，新的色彩版本不仅可以为消费者提供更多的选择余地，而且所带来的新鲜感也同样让人产生购买欲望。比如接下来我们将要介绍的这款索尼爱立信W595c(参数 报价 图片 文章 热评)的黑色版本便是很好的例子。\u3000\u3000从这次亮相的索尼爱立信W595c黑色版的真机来看，该机不同于灰色版本的地方除了色彩上的变化以外，还在Walkman标记及键盘按键上使用了红色进行装饰，因此在整体上看起来更加的赏心悦目。值得一提的是，这个版本的官方名称为“Ruby Black”，直译过来就是“红宝石黑”的意思，结合其黑色机身及红色Walkman标记和按键设计，似乎还真的有些恰如其分的感觉。不过，除了色彩及风格上的变化之外，该版本与过去版本在功能上并没有任何差别。也使用了2.2英寸的QVGA分辨率26万色TFT屏幕，并内置光线感应器和方向感应器，手机可在横向状态下自动切换显示模式。\u3000\u3000而作为一款Walkman手机，索尼爱立信W595自然在音乐功能方面会有特别突出的表现。该机不仅顺应潮流的在手机的首尾两端装载了双扬声器，而且引进了Walkman On Top 快捷音乐操作理念，借助便利的音乐按键，可以快速启动音乐播放功能，以及快速进行诸如播放/暂停/上一曲/下一曲等相关操作。此外，索尼爱立信W595这次也配备了Walkman3.0播放器，兼容多种格式音乐播放和支持FM收音机功能，包括 TrackID 音乐识别，Shake Control 音乐晃动操作 (换曲、调整音量)，SensMe 心情点播器， A2DP 蓝牙立体声，专辑封面显示等功能也是一应俱全。值得一提的是，该机也如高端Walkman手机W980一样也随机搭配了HPM-77立体声耳机，音质效果更为优异。并且除了支持25小时音乐播放以外，也拥有独特的耳机分接头，包括外放音乐或是插入第二组耳机、立体声喇叭都可轻松应对。\u3000\u3000索尼爱立信W595还内置摄像头320万像素，虽然没有提供自动聚焦功能，但主要的拍摄功能还是应有尽有，尤其是BestPic、Photo fix等在Cyber-shot系列手机上的拍照功能在索尼爱立信W595之上的使用，也在一定程度上提升了该机的拍照实力。目前，索尼爱立信W595在配备电池、充电器、HPM-77立体声耳机、2GB容量的M2记忆棒存储卡、USB数据线以及PC套件和Media Manager软件等的情况下，其国内行货的价格在2200-2400元左右，而改版机市场的价格则大约1600元左右。而这次亮相的Ruby Black版本也将会国内上市销售。参考价：2299元上市时间：2008年07月主屏幕：26万色TFT显示屏，2.2英寸外形尺寸：100×47×14mm，65cm3照相功能：320万像素CMOS传感器可选颜色：深蓝色，白色，深灰色，黑色已有_COUNT_位用户评论 点击查看评论推荐想买观望'], 'labels': [1]}
    PROMPTS=[
        # 总结（尝试用于句子生成）
        """summarize this article: {sentence1}""",
        # 改写这段句子
        """rewrite this article: {sentence1}""",
        # 中文文章分类
        """### 指示:下面这篇文章是谁写的?{options_}\n\n### 输入:{sentence1}\n\n### 回答:\n""",
        # prompt_input
        (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        # prompt_no_input
        (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    ]

    sample_data=["为什么要减少污染，保护环境？"]
    
    sources = PROMPTS[4].format_map({'instruction': sample_data})
    logger.info(sources)
    inputs = tokenizer(sources, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    logger.info(input_ids)
    logger.info(input_ids.numpy()[0])

    logger.info("检查tokenizer")
    tokens=tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
    logger.info(f"将 ids 转换成 tokens 文本：ids_to_tokens = {tokens}")
    ids=tokenizer.convert_tokens_to_ids(tokens)
    logger.info(f"将tokens转换成ids：tokens_to_ids = {ids}")

    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=400
    )
    with torch.no_grad():
        generation_output=model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config,
        )
        logger.info(f"generation_output={generation_output}")
    s = generation_output.sequences[0]
    logger.info(f"s={s}")
    output = tokenizer.decode(s, skip_special_tokens=True)
    logger.info(f"output 1={output}")
    output = output.split("### 回答:")[-1].strip()
    logger.info(f"output 2={output}")
