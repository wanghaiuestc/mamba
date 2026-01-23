from transformers import MambaForCausalLM, AutoTokenizer
import torch
from transformers import logging
logging.set_verbosity_info()
# export HF_ENDPOINT=https://hf-mirror.com
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

model.eval()  # inference mode

inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

with torch.inference_mode():  # no grad, faster/less memory
    out = model.generate(**inputs, max_new_tokens=10)

print(tokenizer.batch_decode(out, skip_special_tokens=True))
#pip install sklearn scipy numpy