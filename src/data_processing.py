
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

from datasets import load_dataset
from datasets import DatasetDict

from extract_attention import *
from utils import *
from data_dict import MODEL_DICT

def prepare_data_loader(
    name='imdb',
    split='test',
    model_alias='distilbert',
    model_dict=MODEL_DICT,
    sample_size=None,
    batch_size=16
):
  """ Loads and processes a dataset from the HuggingFace library"""
  dataset = load_dataset(name, split=split)
  if sample_size:
    data_size = len(dataset)
    sample_indices = np.random.choice(
        data_size, size=sample_size, replace=False)
    dataset = dataset.select(sample_indices)

  model_name = model_dict[model_alias]['pretrained_model']
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  dataset = dataset.map(
      lambda batch: tokenizer(
          batch['text'],
          padding='max_length',
          truncation=True,
          return_tensors='pt'
      ),
      batched=True
  )
  dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

  data_loader = DataLoader(dataset, batch_size=batch_size)
  return data_loader
