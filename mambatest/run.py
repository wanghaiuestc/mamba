from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
from transformers import logging
logging.set_verbosity_info()
# export HF_ENDPOINT=https://hf-mirror.com

config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
# add this to avoid problem in transformer 5.0
config.tie_word_embeddings = True
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", config=config)
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", config=config)

model.eval()  # inference mode

inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

with torch.inference_mode():  # no grad, faster/less memory
    out = model.generate(**inputs, max_new_tokens=10)

print(tokenizer.batch_decode(out, skip_special_tokens=True))
#pip install sklearn scipy numpy
