import sagemaker
import boto3
from sagemaker.huggingface import (
	HuggingFace, 
	HuggingFaceModel
)

from typing import Optional, Dict, Union
from fengxai.utils.log_center import create_logger

logger = create_logger()

class HfSagemaker:
	
	def __init__(
        self,
        sagemaker_role_name: str = "sagemaker_ap-southeast-2",
		sagemaker_config: Optional[Dict[str, Union[str, str]]] = None,
		sagemaker_distribution: Optional[Dict] = None,
		env: Optional[Dict[str, Union[str, str]]] = None,
    ):
		default_env = {
			'NCCL_DEBUG': 'INFO',
			'WANDB_API_KEY': "93523e57b94611e1a558a6541f834f17dd400be5",
		}

		logger.info("初始化环境变量")
		self._sagemaker_role_name=sagemaker_role_name
		self._sagemaker_config=sagemaker_config
		if env is not None:
			self._env = {**default_env, **env}
		else:
			self._env = default_env
		self._sagemaker_distribution=sagemaker_distribution

		try:
			self._role = sagemaker.get_execution_role()
		except ValueError:
			logger.info("当前使用local机器进行远程连接；使用boto3.client进行iam权限获取")
			iam_client = boto3.client('iam')
			self._role = iam_client.get_role(RoleName=self._sagemaker_role_name)['Role']['Arn']

	def play(
		self,
		config: Optional[Dict[str, str]] = None,
	):
		# logger.info("可以直接在github下载微调的脚本")
		# git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.17.0'}

		logger.info("构建estimator环境")
		huggingface_estimator = HuggingFace(
			# 本地源文件路径，该源文件路径下文件夹会被发送到 sagemaker training job 机器上执行。
			# 同时 sagemaker 会自动执行该文件夹中的 requirements.txt 文件安装 python 依赖
			# source_dir='./scripts',
			# 将当前目录下所有文件加载到sagemaker用于脚本依赖（注意不要在当前目录下存放大文件，否则传递速度会收到影响，训练会变慢）
			# os.getcwd() 获取当前项目路径
			# source_dir=os.getcwd(),
			source_dir=self._sagemaker_config["source_dir"] if "source_dir" in self._sagemaker_config else "./",
			entry_point=self._sagemaker_config["entry_point"] if "entry_point" in self._sagemaker_config else'auto_train.py',
			# ml.p3.2xlarge  
			# ml.p3.8xlarge 
			# ml.p3dn.24xlarge
			instance_type=self._sagemaker_config["instance_type"] if 'instance_type' in self._sagemaker_config else 'ml.p3.2xlarge',
			instance_count=self._sagemaker_config["instance_count"] if "instance_count" in self._sagemaker_config else 1,
			role=self._role,
			# git_config=git_config,
			image_uri="763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310-ubuntu20.04-sagemaker",
			# transformers_version='4.26.0',
			# pytorch_version='1.13.1',
			py_version='py39',
			hyperparameters=config,
			# 设置 wandb 的key（如果不用wandb可以删除掉该配置项）
			environment=self._env,
			# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html
			distribution=self._sagemaker_distribution,
		)

		logger.info("启动estimator")
		huggingface_estimator.fit(
			job_name="ai-test-v1"
		)


	def deploy(
		self,
		data,
	):
		default_env={
			'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad', 	# model_id from hf.co/models
			# 'HF_TASK':'question-answering',                           # NLP task you want to use for predictions
			'HF_API_TOKEN': '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'
		}
		self._env = {**default_env, **self._env}

		huggingface_model=HuggingFaceModel(
			env=self._env,                                               # configuration for loading model from Hub
			role=self._role,                                        # IAM role with permissions to create an endpoint
			transformers_version='4.26.0',
			pytorch_version='1.13.1',
			py_version='py39',
		)

		# deploy model to SageMaker Inference
		# 将模型部署到 SageMaker 推理
		predictor = huggingface_model.deploy(
			initial_instance_count=1,
			instance_type=self._sagemaker_config["instance_type"] if 'instance_type' in self._sagemaker_config else 'ml.p3.2xlarge',
		)

		# example request: you always need to define "inputs"
		# 请求示例：您始终需要定义“输入”
		# data = {
		# 	"inputs": {
		# 		"question": "What is used for inference?",
		# 		"context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
		# 	}
		# }

		# request
		# 要求
		predictor.predict(data)
