"""
Creates AttentionAnalysis class to run a model and extract attention weights
and gradients.
See usage in '__main__' at the bottom of this file.

Run with memory profiler:
>> python -m memory_profiler extract_attention.py
"""

from transformers import pipeline, set_seed
from transformers import AutoModelForSequenceClassification
from memory_profiler import profile

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from data_processing import prepare_data_loader
from data_dict import MODEL_DICT

class AttentionAnalysis:
  """Class to run a model and extract attention weights and gradients.

  Args:
    model_alias (str): Alias of the model to use, e.g. ['bert', 'distilbert']
    data_loader (torch.utils.data.DataLoader): Dataloader to evaluate on.
      Must have the following keys: ['input_ids', 'attention_mask', 'label']
    model_dict (dict, optional):
      Dictionary containing the model aliases and pretrained models.
      Should be defined in the data_dict.py file. Defaults to MODEL_DICT.
    loss_function ([type], optional): Loss function to use.
      Defaults to nn.CrossEntropyLoss().
  """
  def __init__(
      self,
      model_alias,
      data_loader,
      model_dict=MODEL_DICT,
      loss_function=nn.CrossEntropyLoss()
  ):
    self.model_alias = model_alias
    self.model_name = model_dict[model_alias]['pretrained_model']
    self.data_loader = data_loader
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = AutoModelForSequenceClassification.from_pretrained(
      self.model_name,
      output_attentions=True,
      num_labels=2,
      ignore_mismatched_sizes=True
    ).to(self.device)
    self.model.eval()
    self.loss_function = loss_function
    self.attention_name = model_dict[model_alias]['attention_param_name']
    self.grad_attention_name = model_dict[model_alias]['grad_attention_param_name']
    # Initialize attributes to store results
    self.cls_attention_means = None
    self.cls_attention_grad_means = None
    self.topk_attentions = None
    self.topk_attentions_grad = None
    self.bottomk_attentions_grad = None
    # Store info about the input data:
    self.tokenized_inputs = None
    self.labels = None
    self.predicted_labels = None

  # @profile
  def run(self, top_k=50, store_input_output=False):
    """Run the model and extract attention weights and gradients.

      Args:
        top_k (int): Number of top and bottom attention weights to store.
          Defaults to 200.

      Returns:
        None

      Computes and stores results in class attributes. Key attributes:
        - cls_attention_means: Average attention weights for the CLS token.
        - cls_attention_grad_means: Average attention gradients for the CLS token.
        - topk_attentions: Indices of the top-k attention weights.
        - topk_attentions_grad: Indices of the top-k attention gradients.
        - bottomk_attentions_grad: Indices of the bottom-k attention gradients.
        - tokenized_inputs: Tokenized inputs.
        - labels: True labels.
        - predicted_labels: Predicted labels.
    """
    cls_attention_means = []
    cls_attention_grad_means = []
    topk_attentions = []
    topk_attentions_grad = []
    bottomk_attentions_grad = []
    tokenized_inputs = []
    label_list = []
    predicted_label_list = []

    for batch in self.data_loader:
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      labels = batch['label'].to(self.device)
      label_list.append(labels)

      # Run the model
      output = self.model(input_ids=input_ids, attention_mask=attention_mask)
      predicted_labels = output.logits.argmax(dim=1)
      predicted_label_list.append(predicted_labels.cpu())

      # Do a backward pass to get gradients.
      for attention in output[self.attention_name]:
        attention.retain_grad()
      for attention in output[self.grad_attention_name]:
        attention.retain_grad()
      loss = self.loss_function(output.logits, labels)
      loss.backward(retain_graph=True)

      # Grab all attention heads from the last layer.
      attention = output[self.attention_name][-1]
      # Get the dL/da gradient for each attention head.
      attention_grad = output[self.grad_attention_name][-1].grad
      # Grab attention weights for the CLS token only (index=0). Shape:
      cls_attention = attention[:, :, 0, :]
      cls_attention_grad = attention_grad[:, :, 0, :]
      # Compute average of CLS attention across all attention heads.
      cls_attention_mean = cls_attention.mean(dim=1)
      # Following the paper, compute the grad-weighted average of the attention:
      #   -1 * (dL/da * a)
      # Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9671639
      cls_attention_grad_mean = -(cls_attention_grad * cls_attention).mean(dim=1)
      # Get the indices of the top-k and bottom-k attention weights.
      topk_attention = torch.topk(
          cls_attention_mean, dim=1, k=top_k).indices
      topk_attention_grad = torch.topk(
          cls_attention_grad_mean, dim=1, k=top_k).indices
      bottomk_attention_grad = torch.topk(
          cls_attention_grad_mean, dim=1, k=top_k, largest=False).indices
      # Store results for this batch
      cls_attention_means.append(cls_attention_mean.cpu())
      cls_attention_grad_means.append(cls_attention_grad_mean.cpu())
      topk_attentions.append(topk_attention.cpu())
      topk_attentions_grad.append(topk_attention_grad.cpu())
      bottomk_attentions_grad.append(bottomk_attention_grad.cpu())
      tokenized_inputs.append(batch['input_ids'].cpu())

      del output, attention, attention_grad
      torch.cuda.empty_cache()

    # Store attention results in class attributes:
    self.cls_attention_means = torch.cat(cls_attention_means, dim=0)
    self.cls_attention_grad_means = torch.cat(cls_attention_grad_means, dim=0)
    self.topk_attentions = torch.cat(topk_attentions, dim=0)
    self.topk_attentions_grad = torch.cat(topk_attentions_grad, dim=0)
    self.bottomk_attentions_grad = torch.cat(bottomk_attentions_grad, dim=0)
    # Only store these for small sample datasets:
    if store_input_output:
      # Store info about the input data:
      self.tokenized_inputs = torch.cat(tokenized_inputs, dim=0)
      self.labels = torch.cat(label_list, dim=0)
      self.predicted_labels = torch.cat(predicted_label_list)


if __name__ == '__main__':

  np.random.seed(42)
  torch.manual_seed(42)
  set_seed(42)

  MODEL_ALIAS = 'bart'
  DATA_SIZE = 4
  test_loader = prepare_data_loader(
      MODEL_ALIAS, sample_size=DATA_SIZE, batch_size=16)

  # Usage:
  attn_obj = AttentionAnalysis(MODEL_ALIAS, test_loader)
  attn_obj.run(store_input_output=True)

  print('Model: =======', MODEL_ALIAS, '=======')
  print('cls_attention_means:', attn_obj.cls_attention_means.shape)
  print('topk_attentions:', attn_obj.topk_attentions.shape)
  print('tokenized_inputs:', attn_obj.tokenized_inputs.shape)
