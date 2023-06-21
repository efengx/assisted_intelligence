import torch
import transformers
import copy

from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorWithPadding
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from fengxai.utils.log_center import create_logger
from torch.utils.data import DataLoader
from trl.core import LengthSampler
from trl.trainer import ConstantLengthDataset
from tqdm import tqdm

logger = create_logger()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)
PROMPTS=[
    # 总结（尝试用于句子生成）
    """summarize this article: {sentence1}""",
    # 改写这段句子
    """rewrite this article: {sentence1}""",
    # 中文文章分类
    """### 指示:下面这篇文章是谁写的?\n{options_}\n\n### 输入:{sentence1}\n\n### 回答:""",
    # 用于文章分类
    """Is the following article written by ai or human?\n{options_}\n{sentence1}""",
]
OPTIONS_="""选项:\n0\n1"""

_index_input="text"
_index_output="labels"



class SupervisedDataset(Dataset):
    """格式化监督数据集进行微调."""

    def __init__(
        self, 
        data_args,
        dataset,
        tokenizer: Optional[PreTrainedTokenizer]=None
    ):
        super(SupervisedDataset, self).__init__()
        
        logger.warning("格式化监督数据集...")
        def formatting_dataset(example):
            sources = PROMPTS[2].format_map({'sentence1': example[data_args.index_input], 'options_': OPTIONS_})
            targets = f"{example[data_args.index_output]}{tokenizer.eos_token}"
            return dict(
                sources=sources,
                targets=targets,
            )
        format_dataset = dataset.map(formatting_dataset)
        logger.warning("Tokenizing inputs... This may take some time...")
        train_data_dict = _preprocess(
            format_dataset["sources"], 
            format_dataset["targets"], 
            tokenizer
        )

        self.input_ids = train_data_dict["input_ids"]
        self.labels = train_data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def smart_tokenizer_and_embedding_resize(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"num_new_tokens: {num_new_tokens}")
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return num_new_tokens

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            # longest 填充到批次最长的序列
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # 将 targets 添加到 sources 最后面；将输入和回答进行合并
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    assert len(input_ids) == len(labels)
    return dict(
        input_ids=input_ids, 
        labels=labels
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def _preprocess_data(
    tokenizer: PreTrainedTokenizer,
    dataset,
    data_args,
):
    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(PROMPTS[2], examples[data_args.index_input], examples[data_args.index_output]):
            if input is not None and input !="":
                instruction = instruction+'\n'+input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:tokenizer.model_max_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:tokenizer.model_max_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = dict(
            input_ids=all_input_ids,
            labels=all_labels
        )
        return results
    tokenizer_dataset = dataset.map(
        tokenization,
        batched=True,
        # num_proc=preprocessing_num_workers,
        # remove_columns=["instruction","input","output"],
        keep_in_memory=False,
    )
    return tokenizer_dataset

def _prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = PROMPTS[2].format_map({'sentence1': example[_index_input], 'options_': OPTIONS_})
    return text

def _chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = _prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    data_args,
    model_args,
) -> Dict:
    logger.info("调整分词器特殊令牌，并调整embedding大小")
    smart_tokenizer_and_embedding_resize(
        tokenizer=tokenizer,
        model=model,
    )
    logger.info(tokenizer)

    logger.warning("加载datasest")
    raw_dataset = load_dataset(
        data_args.dataset_name,
        use_auth_token=model_args.hf_hub_token if model_args.use_auth_token and model_args.hf_hub_token is not None else None,
        # 不在内存中复制数据集，降低内存损耗
        keep_in_memory=False
    )
    logger.info(f"加载原始数据集：{raw_dataset}")

    if data_args.max_train_samples is not None:
        raw_dataset["train"]=raw_dataset['train'].select(range(data_args.max_train_samples)) 
    if data_args.max_eval_samples is not None:
        raw_dataset["validation"]=raw_dataset['validation'].select(range(data_args.max_eval_samples))
    if data_args.max_test_samples is not None:
        raw_dataset["test"]=raw_dataset['test'].select(range(data_args.max_test_samples))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 方法一
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        dataset=raw_dataset["train"],
        data_args=data_args,
    )

    # 方法二
    # train_dataset = _preprocess_data(
    #     tokenizer=tokenizer,
    #     dataset=raw_dataset["train"],
    #     data_args=data_args,
    # )

    logger.info(f"train_dataset: {train_dataset}")
    logger.info(train_dataset[:1])

    logger.info("验证tokenized后的形状是否一致")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=2,
        collate_fn=data_collator
    )
    for batch in train_dataloader:
        break
    logger.info(batch)
    logger.info({k: v.shape for k, v in batch.items()})
    logger.info(type(batch))
    logger.info("校验input_ids")
    tokens=tokenizer.convert_ids_to_tokens(batch["input_ids"].numpy()[0])
    logger.info(f"将 ids 转换成 tokens 文本：ids_to_tokens = {tokens}")
    ids=tokenizer.convert_tokens_to_ids(tokens)
    logger.info(f"将tokens转换成ids：tokens_to_ids = {ids}")

    # logger.info("校验labels")
    # logger.info(batch["labels"].numpy())
    # logger.info(batch["labels"].numpy()[0])
    # tokens=tokenizer.convert_ids_to_tokens(batch["labels"].numpy()[0])
    # logger.info(f"将 ids 转换成 tokens 文本：ids_to_tokens = {tokens}")
    # ids=tokenizer.convert_tokens_to_ids(tokens)
    # logger.info(f"将tokens转换成ids：tokens_to_ids = {ids}")

    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        dataset=raw_dataset["validation"],
        data_args=data_args,
    )
    logger.info(f"eval_dataset: {eval_dataset}")

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=2,
        collate_fn=data_collator
    )
    logger.info(f"eval_dataloader: {eval_dataloader}")

    return dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    ), raw_dataset["test"], train_dataloader, eval_dataloader

def maker_data_module(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    data_args,
    model_args,
):
    logger.info("调整分词器特殊令牌，并调整embedding大小")
    smart_tokenizer_and_embedding_resize(
        tokenizer=tokenizer,
        model=model,
    )
    logger.info(tokenizer)

    raw_dataset=load_dataset(
        data_args.dataset_name, 
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token else None,
        # 不在内存中复制数据集，降低内存损耗
        keep_in_memory=False,
    )
    logger.info(f"加载数据集：{raw_dataset}")

    if data_args.max_train_samples is not None:
        raw_dataset["train"]=raw_dataset['train'].select(range(data_args.max_train_samples)) 
    if data_args.max_eval_samples is not None:
        raw_dataset["validation"]=raw_dataset['validation'].select(range(data_args.max_eval_samples))
    if data_args.max_test_samples is not None:
        raw_dataset["test"]=raw_dataset['test'].select(range(data_args.max_test_samples))

    def tokenizer_fun(example):
        return tokenizer(
            example[data_args.index_input],
            # 填充
            # padding="longest",
            # 截取
            truncation=True,
        )
    tokenized_dataset = raw_dataset.map(
        tokenizer_fun, 
        batched=True
    )

    logger.info(tokenized_dataset)
    logger.info(tokenized_dataset["train"][:1])
    logger.info(f"input_ids len = {len(tokenized_dataset['train'][:1]['input_ids'][0])}")
    logger.info(f"attention_mask len = {len(tokenized_dataset['train'][:1]['attention_mask'][0])}")

    hf_train_columns = ["attention_mask", "input_ids", "labels", "token_type_ids"]
    logger.info(f"hf_train_columns: {hf_train_columns}")
    logger.info(f"tokenized_dataset train column: {tokenized_dataset['train'].column_names}")
    
    intersection_list = list(set(hf_train_columns).intersection(set(tokenized_dataset['train'].column_names)))
    logger.info(f"取交集，两者都有：{intersection_list}")

    union_list = list(set(hf_train_columns).union(set(tokenized_dataset['train'].column_names)))
    logger.info(f"取并集，合并两者去掉重复：{union_list}")
    
    difference_list = list(set(tokenized_dataset['train'].column_names).difference(set(intersection_list)))
    logger.info(f"获取差集，只有a有：{difference_list}")

    if len(difference_list) > 0:
        tokenized_dataset["train"] = tokenized_dataset["train"].remove_columns(difference_list)
    logger.info(f"train可用的columns：{tokenized_dataset['train'].column_names}")

    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info("验证tokenized后的形状是否一致")
    train_dataloader = DataLoader(
        tokenized_dataset["train"], 
        shuffle=True, 
        batch_size=2,
        collate_fn=data_collator
    )
    for batch in train_dataloader:
        break
    logger.info(batch)
    logger.info({k: v.shape for k, v in batch.items()})
    logger.info(type(batch))
    tokens=tokenizer.convert_ids_to_tokens(batch.input_ids.numpy()[0])
    logger.info(f"将 ids 转换成 tokens 文本：ids_to_tokens = {tokens}")
    ids=tokenizer.convert_tokens_to_ids(tokens)
    logger.info(f"将tokens转换成ids：tokens_to_ids = {ids}")

    eval_dataloader = DataLoader(
        tokenized_dataset["validation"], 
        shuffle=True, 
        batch_size=2,
        collate_fn=data_collator
    )

    return dict(
        train_dataset=tokenized_dataset["train"], 
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    ), tokenized_dataset["test"], train_dataloader, eval_dataloader

def rlhf_se_dataset(
    tokenizer: PreTrainedTokenizer,
    data_args,
    model_args,
):
    # logger.info("调整分词器特殊令牌，并调整embedding大小")
    # smart_tokenizer_and_embedding_resize(
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    raw_dataset=load_dataset(
        data_args.dataset_name, 
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token else None,
        # 不在内存中复制数据集，降低内存损耗
        keep_in_memory=False,
        # split="train",
    )
    logger.info(f"数据集的字符与标记比率为：{raw_dataset}")

    _index_input = data_args.index_input
    _index_output = data_args.index_output

    if data_args.max_train_samples is not None:
        if isinstance(raw_dataset, DatasetDict):
            raw_train_dataset = raw_dataset["train"]
        if isinstance(raw_dataset, Dataset):
            raw_train_dataset=raw_dataset

        raw_train_dataset=raw_train_dataset.select(range(data_args.max_train_samples))

    chars_per_token = _chars_token_ratio(raw_train_dataset, tokenizer)
    logger.info(f"数据集的平均字符数: {chars_per_token:.2f}")
    logger.info(f"raw_train_dataset={raw_train_dataset}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        raw_train_dataset,
        formatting_func=_prepare_sample_text,
        infinite=True,
        seq_length=tokenizer.model_max_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        raw_dataset["validation"],
        formatting_func=_prepare_sample_text,
        infinite=False,
        seq_length=tokenizer.model_max_length,
        chars_per_token=chars_per_token,
    )

    logger.info("检验dataset数据结构")
    logger.info(train_dataset.dataset[:1])
    example=tokenizer(
        train_dataset.dataset[:1]["text"],
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    logger.info(example);
    
    return dict(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

def rlhf_reward_dataset(
    tokenizer: PreTrainedTokenizer,
    data_args,
    model_args,
    training_args,
):
    raw_dataset=load_dataset(
        data_args.dataset_name, 
        use_auth_token=model_args.hf_hub_token if model_args.hf_hub_token else None,
        # 不在内存中复制数据集，降低内存损耗
        keep_in_memory=False,
        # split="train",
    )

    # Can adjust to be higher if you have more processors.
    num_proc=training_args.num_proc 
    original_columns = raw_dataset.column_names
    _index_input = data_args.index_input
    _index_output = data_args.index_output

    def preprocess_function(example):
        """
        将内容合并
        """
        text = PROMPTS[2].format_map({'sentence1': example[_index_input], 'options_': OPTIONS_})
        return tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    tokenizer_dataset=raw_dataset.map(
        preprocess_function,
        batched=True, 
        return_tensors="pt",
        padding="max_length",
        num_proc=num_proc,
        remove_columns=original_columns
    )
    logger.info(f"step-init: tokenizer_dataset={tokenizer_dataset}")

