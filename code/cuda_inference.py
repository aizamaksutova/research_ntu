import warnings
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel
import torch
import time
import wandb
def fxn():
  warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  fxn()
wandb.login()
wandb.init(project="gpu-inference")
def inference_timing():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  model = BertModel.from_pretrained("bert-large-uncased")
  text = "Replace me by any text you'd like."
  dummy_input = tokenizer(text, return_tensors='pt').to(device)
  model.to(device)
  # initializing
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  repetitions = 300
  timings= np.zeros((repetitions,1))
  #GPU-WARM-UP
  for _ in range(10):
    _ = model(**dummy_input)
  # MEASURE PERFORMANCE
  with torch.no_grad():
    for rep in range(repetitions):
      starter.record()
      _ = model(**dummy_input)
      ender.record()
      # WAIT FOR GPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time
      wandb.log({"current_time": curr_time
         })
  mean_syn = np.sum(timings) / repetitions
  std_syn = np.std(timings)
  return mean_syn, std_syn
res = inference_timing()
print(res)
