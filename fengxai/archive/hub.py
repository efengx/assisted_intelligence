import os
import shutil

from fengxai.utils.log_center import create_logger
from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer
from huggingface_hub import (
    login,
    HfApi,
    create_repo,
)

logger = create_logger()


def _create_inference_script(
    folder_path,
    repo_id,
):
    logger.info(f"folder_path：{folder_path}")
    logger.info(f"repo_id：{repo_id}")
    if not os.path.exists(repo_id):
        os.makedirs(repo_id)

    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, True):
            logger.info(f"转移文件列表：{files}")
            for eachfile in files:
                shutil.copy(os.path.join(root, eachfile), repo_id)

def rlhf_se_save(
    trainer: Trainer,
    training_args,
    model_args,
):
    logger.info("step-save: 保存模型")
    login(token=model_args.write_hf_hub_token, add_to_git_credential=True)

    # model.save_pretrained(training_args.clone_repo)
    # 存在超参数格式问题 AttributeError: 'str' object has no attribute 'value' 
    # for hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    # trainer.create_model_card()

    if isinstance(trainer, SFTTrainer):
        logger.info("SFTTrainer保存")
        logger.info("Saving last checkpoint of the model")
        trainer.model.save_pretrained(training_args.clone_repo)
        trainer.tokenizer.save_pretrained(training_args.clone_repo)
        # trainer.model.push_to_hub(
        #     training_args.clone_repo, 
        #     use_auth_token=model_args.write_hf_hub_token,
        #     use_temp_dir=False,
        # )
    else:
        logger.info("Trainer保存")
        trainer.save_model(training_args.clone_repo)
    logger.info(f"保存模型到本地[{training_args.clone_repo}]目录")
    
    logger.info("step-upload(huggingface): 上传模型到hub")
    hf_host="https://huggingface.co"
    # 上传最新的模型到存储库
    api=HfApi()
    list_models=api.list_models(
        search=training_args.clone_repo, 
        token=model_args.hf_hub_token
    )
    if len(list_models) == 0:
        # 创建远程存储库（后续判断是否需要？）
        repo_url=create_repo(
            training_args.clone_repo, 
            private=True,
            token=model_args.write_hf_hub_token
        )
        logger.info(f"创建远程存储库: {repo_url}")
    else:
        logger.info(list_models)
        logger.info(list_models[0])

    # 将inference文件复制到存储库
    # _create_inference_script(
    #     folder_path=os.path.join(os.getcwd(), f'fengx_ai/inference/code/{training_args.training_model}'),
    #     repo_id=os.path.join(training_args.clone_repo, 'code')
    # )

    # 上传目录下的文件到存储库
    logger.info(f"folder_path: {training_args.clone_repo}")
    list_dir=os.listdir(training_args.clone_repo)
    logger.info(f"check: 需要上传的文件目录为：{list_dir}")
    api.upload_folder(
        folder_path=training_args.clone_repo,
        repo_id=training_args.clone_repo,
        token=model_args.write_hf_hub_token
    )
    # 查询存储库并打印(内容)
    logger.info(f"step-hub: {hf_host}/{training_args.clone_repo}")

def model_save(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_args,
    model_args,
):
    logger.info(f"保存模型到：{training_args.clone_repo}")
    # 方法一：
    # logger.info("保存lora模型和分词器")
    login(token=model_args.write_hf_hub_token, add_to_git_credential=True)
    tokenizer.save_pretrained(training_args.clone_repo, push_to_hub=True)
    model.save_pretrained(training_args.clone_repo, push_to_hub=True)
    # 存在超参数格式问题 AttributeError: 'str' object has no attribute 'value' 
    # for hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    # trainer.create_model_card()

    # 方法二：
    # 保存模型(根据 trainer 中传入的 tokenizer 会自动保存 tokenizer)
    # trainer.save_model(training_args.clone_repo)
    

    # logger.info("******* 上传新训练的模型到 huggingface hub ********")
    # hf_host="https://huggingface.co"
    # # 上传最新的模型到存储库
    # logger.info("*** upload model hub ***")
    # api=HfApi()
    # list_models=api.list_models(
    #     search=training_args.clone_repo, 
    #     token=model_args.hf_hub_token
    # )
    # if len(list_models) == 0:
    #     # 创建远程存储库（后续判断是否需要？）
    #     create_repo(
    #         training_args.clone_repo, 
    #         private=True,
    #         token=model_args.write_hf_hub_token
    #     )
    # else:
    #     logger.info(list_models[0])

    # # 将inference文件复制到存储库
    # _create_inference_script(
    #     folder_path=os.path.join(os.getcwd(), f'fengx_ai/inference/code/{training_args.training_model}'),
    #     repo_id=os.path.join(training_args.clone_repo, 'code')
    # )

    # # 上传目录下的文件到存储库
    # logger.info(f"folder_path:{training_args.clone_repo}")
    # list_dir=os.listdir(training_args.clone_repo)
    # logger.info(list_dir)
    # api.upload_folder(
    #     folder_path=training_args.clone_repo,
    #     repo_id=training_args.clone_repo,
    #     token=model_args.write_hf_hub_token
    # )
    # # 查询存储库并打印(内容)
    # logger.info(f"{hf_host}/{training_args.clone_repo}")
