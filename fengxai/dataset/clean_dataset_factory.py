import sys
import os
import boto3
import aiobotocore.session

from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional
from datetime import datetime
# 可以是 pytorch 模型
from torchview import draw_graph
from fengxai.utils.log_center import create_logger
from fengxai.config.clean_dataset_config import CleanDatasetArguments
from fengxai.utils.calculate import intersection

logger = create_logger()



"""
flan-t5 instructions prompts，
sentence1 是 input,
target 是 output,
数据结构：
column=[id, sentence1, target]
"""
prompts=[
    # 总结（尝试用于句子生成）
    """summarize this article: {sentence1}""",
    # 改写这段句子
    """rewrite this article: {sentence1}""",
    # 中文文章分类
    """下面这篇文章是AI写的还是人类写的?{options_}。{sentence1}""",
    # 用于文章分类
    """Is the following article written by ai or human?\n{sentence1}\n{options_}""",
]

# options_="""OPTIONS:\n-ai write \n-human write """
options_="""选项:ai写的,人类写的"""

# flan-t5 关键参数
# 设置训练最大的输入长度。更大的长度会导致内存的需求呈现指数级增长
parameters={
    "max_input_length": 29,
    "max_target_length": 64,
    "batch_size": 32,
    # "num_return_sequences": 3,
    # "top_k": 50,
    # "top_p": 0.95,
    # "do_sample": True,
}

os.environ["HF_TOKEN"]="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"

class CleanDatasetFactory:
    
    def __init__(
            self,
            input_path: Optional[str] = None,
            clean_dataset_args: CleanDatasetArguments=None,
        ):
        self.input_path=input_path
        self.clean_dataset_args=clean_dataset_args


    def _callback_preprocess_function(
            self,
            examples,
        ):
        """
        预处理模型的输入
        """
        # 将 instriction head 添加到文本中
        for text in examples["sentence1"]:
            input_text=prompts[2].replace("{sentence1}", text)
            input_text=input_text.replace("{options_}", options_)
        
        encoding_input=self.tokenizer(
            input_text, 
            # 存在错误：ArrowInvalid: Column 2 named input_ids expected length 1000 but got length 512
            # max_length=parameters["max_input_length"], 
            # truncation=True：截断为模型可接受的最大输入长度。
            truncation=True,
            # 当句子长度小于 max_length 时，自动补全
            padding="max_length",
            # 设置旧列到新列之间的映射
            # return_overflowing_tokens=True
        )

        for target in examples["target"]:
            target_input=target
        
        # Setup the tokenizer for targets
        # 为目标设置分词器
        with self.tokenizer.as_target_tokenizer():
            labels=self.tokenizer(
                target_input, 
                # max_length=parameters["max_target_length"], 
                # truncation=True：截断为模型可接受的最大输入长度。
                truncation=True,
                # 当句子长度小于 max_length 时，自动补全
                padding="max_length"
            )

        encoding_input["labels"]=labels["input_ids"]

        # 将旧列复制到新列，防止 batched 转换错误
        # logger.info(model_inputs)
        # sample_map=encoding_input.pop("overflow_to_sample_mapping")
        # for key, values in examples.items():
        #     encoding_input[key] = [values[i] for i in sample_map]
        return encoding_input


    def _check_token(
            self,
            example
        ):
        """
        查看token的特征
        """
        # 将 instriction head 添加到文本中
        for text in example:
            input_text=prompts[2].replace("{sentence1}", text)
            input_text=input_text.replace("{options_}", options_)
        encoding_input=self.tokenizer(
            input_text, 
            # 存在错误：ArrowInvalid: Column 2 named input_ids expected length 1000 but got length 512
            # max_length=parameters["max_input_length"], 
            # truncation=True：截断为模型可接受的最大输入长度。
            truncation=True,
            # 当句子长度小于 max_length 时，自动补全
            # padding="max_length",
            padding=True,
            # 设置旧列到新列之间的映射
            # return_overflowing_tokens=True,
            return_tensors="pt",
        )

        logger.info(encoding_input)
        # logger.info(encoding_input.input_ids)
        # logger.info(self.tokenizer.batch_decode(encoding_input.input_ids))
        logger.info(encoding_input.input_ids.numpy())
        logger.info(encoding_input.input_ids.numpy().shape)
        logger.info(encoding_input.input_ids.numpy().ndim)
        # if encoding_input.input_ids.numpy().ndim == 2:
        #     logger.info(self.tokenizer.convert_ids_to_tokens(encoding_input.input_ids.numpy()[0]))

        # 将 encoding_input 转换回文本
        tokens=self.tokenizer.convert_ids_to_tokens(encoding_input.input_ids.numpy()[0])
        logger.info(tokens)
        # 将文本转换成ids
        ids=self.tokenizer.convert_tokens_to_ids(tokens)
        logger.info(ids)

        # 将 id 转换成可显示的令牌
        # （只能应用与慢速分词器上，在快速分词器上不可用）
        # logger.info(encoding_input.tokens())
        # logger.info(len(encoding_input.tokens()))
        # 查看单词的索引
        # logger.info(encoding_input.word_ids())


    def to_clean(
            self,
            file_name_list: List[str] = None,
            is_save: bool=False,
            push_to_hub: str=None,
            number: str=None,
            human_zoom: str=1,
        ):
        """
        清洗数据
        """
        # dataset=load_dataset(
        #     self.input_path,
        #     data_files=file_name_list,
        #     column_names=['label', 'text'],
        #     split= "train",
        #     skiprows=1
        # )
        # 第一个数据集
        ai_dataset=load_dataset(
            self.input_path,
            data_files=file_name_list[0],
            column_names=['id', 'text'],
            split= "train",
            skiprows=1
        )

        # 第二个数据集
        # 调整输入列的标准格式为：text
        human_dataset=load_dataset(
            # "datasets/textclass/artificial_article",
            # data_files="output-thucnews-162929.csv",
            self.input_path,
            data_files=file_name_list[1],
            column_names=['id', 'text'],
            split= "train",
            skiprows=1
        )

        # 修改标签的值
        ai_dataset=ai_dataset.map(
            lambda _: {"target": "ai写的"}, 
            # 删除多余的id列
            remove_columns=["id"]
        )
        # 修改 target 的类型为 string
        # one_dataset.cast_column("target", Value("string"))
        logger.info(ai_dataset)
        logger.info(f"ai raw dataset len: {len(ai_dataset)}")

        # 修改标签的值
        human_dataset=human_dataset.map(
            lambda _: {"target": "人类写的"},
            # 删除多余的id列
            remove_columns=["id"]
        )
        # 修改 target 的类型为 string
        # one_dataset.cast_column("target", Value("string"))

        # 设置默认的长度
        if number is not None and number < len(ai_dataset):
            ai_dataset=ai_dataset.select(range(number))

        ai_dataset_size=len(ai_dataset)
        human_dataset_size=len(human_dataset)
        # 均衡两个数据集，使两种类型数量一致
        if ai_dataset_size < human_dataset_size:
            human_dataset=human_dataset.select(range(ai_dataset_size * human_zoom))
        else:
            ai_dataset=ai_dataset.select(range(human_dataset_size))

        # 合并数据集
        dataset=concatenate_datasets([ai_dataset, human_dataset])
        logger.info(dataset)

        # 数据集拆分成3份
        main_dataset = dataset.shuffle(seed=5)                           # 随机乱序，用于后续的拆分
        
        train_test = main_dataset.train_test_split(test_size=0.2)
        test_valid = train_test["test"].train_test_split(test_size=0.5)

        dataset_dict = DatasetDict({
            # 训练集
            'train': train_test["train"],
            # 验证集（用于训练过程中损失函数的验证）
            'validation': test_valid["train"],
            # 测试集（用于测试泛化过程）
            'test': test_valid["test"],
        })

        # 剔除无效内容
        # def compute_text_size(example):
        #     return {
        #         "text_size": len(example["sentence1"].split()), 
        #         "text_length": len(example["sentence1"])
        #     }
        # dataset_dict=dataset_dict.map(compute_text_size)

        """
        保存到csv
        datasetdict 和 dataset 会有不同的本地化方式
        """
        save_dataset=dataset_dict
        if is_save:
            if dataset_dict is not None:
                # 保存 dataset_dict 类型
                dataset_dict.save_to_disk(f"{self.input_path}/output/train-{len(dataset)}")
                save_dataset=dataset_dict
            else:
                # 覆盖保存到csv会覆盖同名文件(只适用于dataset)
                dataset.to_csv(f"{self.input_path}/output/train-{len(dataset)}.csv")
                save_dataset=dataset

        logger.info("保存到hub")
        if push_to_hub is not None:
            # login(token=None, add_to_git_credential=True)
            save_dataset.push_to_hub(
                push_to_hub, 
                private=True,
                token=self.clean_dataset_args.write_hf_hub_token
            )
        
        # 查看dataset示例
        logger.info(save_dataset)
        # 查看前2个示例
        logger.info(save_dataset["train"][:2])
        return save_dataset

    def to_preprocessing(
            self,
            model_name,
            dataset_name,
            dataset_name_subset=None,
        ):
        """
        数据预处理
        """
        # load imdb with datasets
        # 加载 datasets
        if dataset_name_subset is None:
            dataset=load_dataset(dataset_name, use_auth_token=os.environ.get('HF_TOKEN'))
        else:
            dataset=load_dataset(dataset_name, dataset_name_subset)
        
        # 显示 dataset
        show_dataset=dataset
        # 将数据格式化转换成 pandas 用于可视化显示
        show_dataset.set_format("pandas")
        # 查看数据示例
        logger.info(show_dataset)
        logger.info(show_dataset["train"][:2])
        
        # 分词器-特征识别(旧的分词器)
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 测试预处理函数
        # result=self._callback_preprocess_function(examples=dataset["train"][0])
        # logger.info(result)
        # logger.info([len(inp) for inp in result["input_ids"]])

        # TODO: 查看可显示的特征
        self._check_token(example=dataset["train"][0]["sentence1"])

        # 检测清洗结果
        # tokenized_dataset=dataset.map(
        #     self._callback_preprocess_function, 
        #     # 开启多线程处理
        #     batched=True
        #     # 当 batched=True是，num_proc设置开启的最大线程数量
        #     # num_proc=8,
        # )
        
        # # 显示数据
        # show_tokenized_dataset=tokenized_dataset
        # # 将数据格式转换成 pandas 用于可视化显示
        # show_tokenized_dataset.set_format("pandas")
        # logger.info(show_tokenized_dataset)
        # logger.info(show_tokenized_dataset["train"][:2])
        # return tokenized_dataset

    def visualize(
            self,
            model_name,
            use_auth_token=False,
        ):
        # 可视化模型
        # 加载模型
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=False if not use_auth_token else os.environ.get('HF_TOKEN')
        )
        self.model=AutoModel.from_pretrained(
            model_name,
            use_auth_token=os.environ.get('HF_TOKEN')
        )

        inputs=self.tokenizer(
            """
            山脉高耸，云雾缭绕，如同一座天然的屏障，将大地分割成不同的区域。
            它们不仅给人们带来美丽的景色，更是自然界中最壮观的存在之一。
            在山间徜徉，感受那种无与伦比的宁静与神秘，仿佛置身于仙境之中。  
            山脉的高度和峻峭，常常令人望而生畏。
            但是，正是这种峻峭，才赋予山脉与众不同的魅力。
            在峰巅之上，云海缭绕，让人有一种登临天空的感觉。
            站在山巅，俯瞰着整个山脉，感受着身心的完美释放，让人心旷神怡。  
            山脉中有着丰富的生态系统，许多珍稀濒危的动植物都生长在这里。山中清澈的溪水，如同一条条银带，在山间蜿蜒流淌。
            山脉的生态环境是人们探险与征服的对象，也是蕴含着自然奥秘的宝库。  
            山脉是人们探险和征服的目标，也是人们心灵的净土。攀登山峰，是一种挑战自我，
            超越极限的经历。在攀登过程中，身体和意志得到了磨砺和锻炼。
            攀登成功后的那一刻，是最让人感到自豪的时刻。
            """, 
            return_tensors="pt"
        )
        model_graph=draw_graph(
            self.model, 
            input_data=inputs,
            # 需要安装 brew install graphviz 包
            save_graph=True,
        )
        model_graph.visual_graph

        # print(model_graph.visual_graph)

    def visualize_tensorboard(
            self,
            model_name,
            use_auth_token=False,
        ):
        # 可视化模型
        # 加载模型
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=False if not use_auth_token else os.environ.get('HF_TOKEN')
        )
        self.model=AutoModel.from_pretrained(
            model_name,
            use_auth_token=os.environ.get('HF_TOKEN')
        )

        inputs=self.tokenizer(
            """
            山脉高耸，云雾缭绕，如同一座天然的屏障，将大地分割成不同的区域。
            它们不仅给人们带来美丽的景色，更是自然界中最壮观的存在之一。
            在山间徜徉，感受那种无与伦比的宁静与神秘，仿佛置身于仙境之中。  
            山脉的高度和峻峭，常常令人望而生畏。
            但是，正是这种峻峭，才赋予山脉与众不同的魅力。
            在峰巅之上，云海缭绕，让人有一种登临天空的感觉。
            站在山巅，俯瞰着整个山脉，感受着身心的完美释放，让人心旷神怡。  
            山脉中有着丰富的生态系统，许多珍稀濒危的动植物都生长在这里。山中清澈的溪水，如同一条条银带，在山间蜿蜒流淌。
            山脉的生态环境是人们探险与征服的对象，也是蕴含着自然奥秘的宝库。  
            山脉是人们探险和征服的目标，也是人们心灵的净土。攀登山峰，是一种挑战自我，
            超越极限的经历。在攀登过程中，身体和意志得到了磨砺和锻炼。
            攀登成功后的那一刻，是最让人感到自豪的时刻。
            """, 
            return_tensors="pt"
        )

        # 创建一个tensorboard日志
        writer = SummaryWriter('runs/fashion_mnist_experiment_1')
        # 查看模型的属性
        self.model.parameters()
        writer.add_graph(self.model, inputs)
        writer.close()
        
    def view_csv(
            self,
            file_name_list: List[str],
        ):
        dataset=load_dataset(
            self.input_path,
            data_files=file_name_list[0],
            split= "train",
            skiprows=1
        )
        logger.info(dataset)
        logger.info(dataset.shape)
        logger.info(dataset["train"][:3])

    def save_to_s3(
            self,
            bucket_s3_path
        ):
        # 获取本地文件
        dataset=load_from_disk(
            self.input_path
        )
        logger.info(dataset)
        
        # 保存到s3
        AWS_REGION = "us-west-2"
        client = boto3.client("s3", region_name=AWS_REGION)
        response = client.list_buckets()
        logger.info(response)

        s3_session = aiobotocore.session.AioSession(profile="default")
        logger.info(s3_session)
        storage_options = {
            "session": s3_session
        }
        # 问题：botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the CreateMultipartUpload operation: Access Denied
        # 说明：aiobotocore 只能适配特定版本的 boto3 需要采用内连安装的方式，否则会出现接口调用异常 
        # 解决方案：pip install -U 'aiobotocore[awscli,boto3]'
        dataset.save_to_disk(
            bucket_s3_path,
            storage_options=storage_options
        )

    def save_to_hf(
            self, 
            dataset_dict,
            is_save_disk=False
        ):
        """
        保存到csv
        datasetdict 和 dataset 会有不同的本地化方式
        """
        save_dataset=dataset_dict
        if is_save_disk:
            if dataset_dict is not None:
                # 保存 dataset_dict 类型
                dataset_dict.save_to_disk(f"{self.input_path}/output/train-{len(dataset_dict)}")
                save_dataset=dataset_dict
            else:
                # 覆盖保存到csv会覆盖同名文件(只适用于dataset)
                dataset_dict.to_csv(f"{self.input_path}/output/train-{len(dataset_dict)}.csv")
                save_dataset=dataset_dict

        logger.info("保存到hub")
        if self.clean_dataset_args.clone_repo is not None:
            # login(token=None, add_to_git_credential=True)
            save_dataset.push_to_hub(
                self.clean_dataset_args.clone_repo, 
                private=True,
                token=self.clean_dataset_args.write_hf_hub_token
            )
        
        # 查看dataset示例
        logger.info(save_dataset["train"][:2])
        return save_dataset

    def merge_and_split_dataset(self):
        self.input_path = self.clean_dataset_args.input_path

        # suffix_list = [".txt", ".csv", ".xlsx"]
        suffix_list = [".csv"]

        for index, file_name in enumerate(self.clean_dataset_args.file_name_list):
            output_title = None

            if file_name.endswith(tuple(suffix_list)):
                dataset=load_dataset(
                    self.input_path,
                    data_files=file_name,
                    split= "train",
                    # skiprows=1
                )
            else:
                dataset=load_dataset(file_name, split="train", revision= "validation")
            logger.info(f"index： {index}，file_name：{file_name}，dataset：{dataset}")
            logger.info(dataset[:1])

            input_title_list = intersection(self.clean_dataset_args.input_title_list, dataset.column_names)
            logger.info(input_title_list)

            if len(input_title_list) > 0:
                input_title = input_title_list[len(input_title_list)-1]
            else:
                logger.warning("input title 没有输入匹配项，将采用默认配置 text")
                input_title = "text"

            output_title_list = intersection(self.clean_dataset_args.output_title_list, dataset.column_names)
            logger.info(f"并集为，两者都有：{output_title_list}")

            if len(output_title_list) > 0:
                output_title = output_title_list[len(output_title_list)-1]
            else:
                logger.warning("output title 没有输入匹配项，将采用默认配置 label")
            
            if output_title is not None:
                dataset = dataset.select_columns([input_title, output_title])
            else:
                dataset = dataset.select_columns([input_title])
                logger.info("添加label值为 file_name_list的索引")
                output_title = "labels"
                dataset=dataset.map(
                    lambda _: {"labels": index}
                )

            dataset = dataset.rename_columns({input_title: "text", output_title: "labels"})

            logger.info(f"dataset: {dataset}")
            logger.info(dataset[:1])

            if 'merge_dataset' in locals():
                merge_dataset = concatenate_datasets([merge_dataset, dataset])
            else:
                merge_dataset = dataset

        # 数据集拆分成3份
        main_dataset = merge_dataset.shuffle(seed=5)                           # 随机乱序，用于后续的拆分
        
        train_test = main_dataset.train_test_split(test_size=0.2)
        test_valid = train_test["test"].train_test_split(test_size=0.5)

        dataset_dict = DatasetDict({
            # 训练集
            'train': train_test["train"],
            # 验证集（用于训练过程中损失函数的验证）
            'validation': test_valid["train"],
            # 测试集（用于测试泛化过程）
            'test': test_valid["test"],
        })

        if self.clean_dataset_args.train_dataset_lenght is not None:
            seed=42
            dataset_dict["train"]=dataset_dict['train'].shuffle(seed=seed).select(range(self.clean_dataset_args.train_dataset_lenght)) 
            dataset_dict["validation"]=dataset_dict['validation'].shuffle(seed=seed).select(range(self.clean_dataset_args.train_dataset_lenght//2))
            dataset_dict["test"]=dataset_dict['test'].shuffle(seed=seed).select(range(self.clean_dataset_args.train_dataset_lenght//2))

        logger.info(f"dataset_dict： {dataset_dict}")
        logger.info(dataset_dict["train"][:1])
        return dataset_dict
   

    def prompt_merge_and_split_dataset(self):
        self.input_path = self.clean_dataset_args.input_path

        # suffix_list = [".txt", ".csv", ".xlsx"]
        suffix_list = [".csv"]

        for index, file_name in enumerate(self.clean_dataset_args.file_name_list):
            output_title = None

            if file_name.endswith(tuple(suffix_list)):
                dataset=load_dataset(
                    self.input_path,
                    data_files=file_name,
                    split= "train",
                    # skiprows=1
                )
            else:
                dataset=load_dataset(file_name, split="test")
            logger.info(f"index： {index}，file_name：{file_name}，dataset：{dataset}")
            logger.info(dataset[:1])

            input_title_list = intersection(self.clean_dataset_args.input_title_list, dataset.column_names)
            logger.info(input_title_list)

            if len(input_title_list) > 0:
                input_title = input_title_list[len(input_title_list)-1]
            else:
                logger.warning("input title 没有输入匹配项，将采用默认配置 text")
                input_title = "text"

            output_title_list = intersection(self.clean_dataset_args.output_title_list, dataset.column_names)
            logger.info(f"并集为，两者都有：{output_title_list}")

            if len(output_title_list) > 0:
                output_title = output_title_list[len(output_title_list)-1]
            else:
                logger.warning("output title 没有输入匹配项，将采用默认配置 label")
            
            if output_title is not None:
                dataset = dataset.select_columns([input_title, output_title])
            else:
                dataset = dataset.select_columns([input_title])
                logger.info("添加label值为 file_name_list的索引")
                output_title = "labels"
                dataset=dataset.map(
                    lambda _: {"labels": index}
                )

            dataset = dataset.rename_columns({input_title: "input", output_title: "label"})

            logger.info(f"dataset: {dataset}")
            logger.info(dataset[:1])

            def filter_dataset_len(example):
                input=None
                raw_inputs=None

                if file_name == "multi_news" and '|||||' in example["input"]:
                    # logger.info(f"{file_name}数据集采用|||||符号对文本进行分割")
                    raw_inputs=example["input"].split("|||||")
                elif '\u3000\u3000' in example["input"]:
                    # logger.info(f"{file_name}数据集采用\\u3000\\u3000符号对文本进行分割")
                    raw_inputs=example["input"].split('\u3000\u3000')
                elif '\n\n' in example["input"]:
                    # logger.info(f"{file_name}数据集采用\\n\\n符号对文本进行分割")
                    raw_inputs=example["input"].split('\n\n')
                elif '  ' in example["input"]:
                    # logger.info(f"{file_name}数据集采用'  '符号对文本进行分割")
                    raw_inputs=example["input"].split('  ')
                
                if raw_inputs is not None:
                    for raw_input in raw_inputs:
                        if len(raw_input) <= self.clean_dataset_args.input_max_length:
                            tmp_input=raw_input if input is None else input + "\n" + raw_input
                            if len(tmp_input) > self.clean_dataset_args.input_max_length:
                                break
                            else:
                                input=tmp_input
                    example["input"]=input
                return example
            dataset = dataset.map(filter_dataset_len)
            dataset = dataset.filter(lambda x: x["input"] is not None and len(x["input"]) <= self.clean_dataset_args.input_max_length)
            logger.info(f"filter <= {self.clean_dataset_args.input_max_length}之后的数量为：\n{dataset}")
            logger.info(dataset[:1])
            logger.info(len(dataset[:1]["input"][0]))            

            if 'merge_dataset' in locals():
                merge_dataset = concatenate_datasets([merge_dataset, dataset])
            else:
                merge_dataset = dataset

        # 修改数据集结构
        def prompt_format(example):
            # example["instruction"]="Classify the following texts as AI write or human write."
            example["instruction"]="将以下文本归类为AI编写还是人类编写？"
            if example["label"] == 0:
                # example["output"]="AI write"
                example["output"]="AI编写"
            elif example["label"] == 1:
                # example["output"]="human write"
                example["output"]="人类编写"
            return example
        merge_dataset=merge_dataset.map(
            prompt_format,
            remove_columns=["label"]
        )

        # 数据集拆分成3份
        main_dataset = merge_dataset.shuffle(seed=5)                           # 随机乱序，用于后续的拆分
        
        train_test = main_dataset.train_test_split(test_size=0.2)
        test_valid = train_test["test"].train_test_split(test_size=0.5)

        dataset_dict = DatasetDict({
            # 训练集
            'train': train_test["train"],
            # 验证集（用于训练过程中损失函数的验证）
            'validation': test_valid["train"],
            # 测试集（用于测试泛化过程）
            'test': test_valid["test"],
        })

        if self.clean_dataset_args.train_dataset_lenght is not None:
            dataset_dict["train"]=dataset_dict['train'].select(range(self.clean_dataset_args.train_dataset_lenght)) 
            dataset_dict["validation"]=dataset_dict['validation'].select(range(self.clean_dataset_args.train_dataset_lenght//2))
            dataset_dict["test"]=dataset_dict['test'].select(range(self.clean_dataset_args.train_dataset_lenght//2))

        logger.info(f"dataset_dict： {dataset_dict}")
        logger.info(dataset_dict["train"][:1])
        return dataset_dict
   

    def up_dataset_task(self):
        logger.info(f"step-task: {self.clean_dataset_args.task_name}")
        if self.clean_dataset_args.task_name == "MERGE_SPLIT_SAVE":
            dataset_dict = self.merge_and_split_dataset()
            if self.clean_dataset_args.is_save_hf:
                save_dataset = self.save_to_hf(dataset_dict=dataset_dict)
                logger.info(save_dataset)
        elif self.clean_dataset_args.task_name == "PROMPT_MERGE_SPLIT_SAVE":
            dataset_dict = self.prompt_merge_and_split_dataset()
            if self.clean_dataset_args.is_save_hf:
                save_dataset = self.save_to_hf(dataset_dict=dataset_dict)
                logger.info(save_dataset)
        else:
            raise ValueError(
                "无效的任务名称，请先定义有效的任务名称。可用的值为：--task_name TO_CLEAN"
            )


if __name__ == '__main__':

    DatasetFactory=CleanDatasetFactory(
        input_path="~/Desktop/ai/fengx_dataset/input/text-classification"
    )
    # fxDatasetFactory.to_clean(
    #     file_name_list=["ai20230420-汇总.csv", "artificial-thucnews-162929-汇总.csv"],
    #     is_save=False,
    #     push_to_hub="rjx/ai-and-human-20",
    #     number=10,
    # )

    # 包含指令的数据集
    # 0516: ai=30169 human=60338 sum=90507 train=72405 validation=9051 test=9051
    DatasetFactory.to_clean(
        file_name_list=["ai20230516-汇总.csv", "artificial-thucnews-162929-汇总.csv"],
        is_save=False,
        push_to_hub="rjx/ai-and-human-0516",
        human_zoom=2,
    )

    # DatasetFactory.to_preprocessing(
    #     model_name="google/flan-t5-base",
    #     dataset_name="rjx/ai-and-human"
    # )

    # DatasetFactory.to_preprocessing(
    #     model_name="THUDM/chatglm-6b",
    #     dataset_name="rjx/ai-and-human"
    # )

    # DatasetFactory.to_preprocessing(
    #     model_name="THUDM/chatglm-6b",
    #     dataset_name="rjx/ai-and-human"
    # )

    # DatasetFactory.visualize(
    #     model_name="rjx/rjxai-albert-v2"
    # )

    # DatasetFactory.visualize(
    #     model_name="ziqingyang/chinese-llama-lora-7b"
    # )

    # DatasetFactory=FxDatasetFactory(
    #     input_path="/Volumes/Thunderbolt/ofengx/ai/fengx_ai_llama_chinese/notebook"
    # )
    # DatasetFactory.view_csv(
    #     file_name_list=["pCLUE_train.csv"],
    # )

    # DatasetFactory=FxDatasetFactory(
    #     input_path="fengx_dataset/output/train-46474/train"
    # )
    # DatasetFactory.save_to_s3(
    #     bucket_s3_path="s3://sagemaker-us-west-2-249450752701/datasets/ai-and-human/train"
    # )

    # DatasetFactory=FxDatasetFactory(
    #     input_path="fengx_dataset/output/train-46474/validation"
    # )
    # DatasetFactory.save_to_s3(
    #     bucket_s3_path="s3://sagemaker-us-west-2-249450752701/datasets/ai-and-human/validation"
    # )
