import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fengxai.utils.log_center import create_logger

logger = create_logger()

logger.info("step-check: 检查pip依赖")
import pip
logger.info(pip.main(['list']))

model = "tiiuae/falcon-7b-instruct"
# model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    logger.info(f"Result: {seq['generated_text']}")
