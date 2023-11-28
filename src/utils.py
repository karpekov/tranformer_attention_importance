"""
Utility functions to print and inspect the results of the attention analysis.
See usage in '__main__' at the bottom of this file.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from transformers import set_seed

from extract_attention import *
from data_processing import *
from data_dict import MODEL_DICT
from extract_attention import AttentionAnalysis

def print_attention_weights(tokens, attention_weights):
  """For tokens and their attention weights, print each them out on new line.

  Args:
    tokens (list): List of tokens.
      You can access it like this from the tokenized_input:
        - tokenized_input[INDEX].tokens
    attention_weights (torch.Tensor): 1D Tensor of attention weights.
  Returns:
    Prints to stdout.

  Sample Output:
    Token       | Attention Weights
    ----------------------------------
    [CLS]       | 0.0432   xx
    this        | 0.0705   xxx
    movie       | 0.0224   x
    was         | 0.1243   xxxxxx
    complete    | 0.1171   xxxxxx
    trash       | 0.1623   xxxxxxxx
    ...
  """
  # Create a bar chart using "x" ranging from 0 x's to 10 x's next to each token
  # based on the relative weight of each token
  attention_weights_x = (
      10*(attention_weights / (attention_weights.max() - attention_weights.min()))).int()

  print(f"{'Token':16}| Attention Weights\n{'-'*40}")
  for i in range(len(tokens)):
    # Skip PAD tokens.
    if tokens[i] != '[PAD]':
      print(
          f"{tokens[i]:16}| {attention_weights[i]:.4f}   {'x'*attention_weights_x[i]}")


def attention_weights_to_df(tokens, attention_weights):
  """Convert tokens and attention weights into DataFrame and display with colored bars.

  Args:
    tokens (List[str]): List of tokesn.
    attention_weights (torch.Tensor): 1D Tensor of attention weights.

  Returns:
    df (pandas.DataFrame): Dataframe with colored bars.
  """

  df = (pd.DataFrame({
      'tokens': tokens,
      'attention': attention_weights.tolist()
  })
      .query('tokens != "PAD"')
      .style
      # Red, green
      .bar(subset=['attention'], align='mid', color=['#d65f5f', '#5fba7d'])
  )
  return df


def get_top_token_list(tokenized_input, top_tokens_att, i):
  """Keep tokens whose indices are listed in the topk list.

  Args:
    tokenized_input (List[List[str]]]): List of word tokens for each input.
    top_tokens_att (torch.Tensor): 2D tensor of top tokens.
    i (int): Index of example in batch.

  Returns:
    top_token_list (list): List of tokens.
  """
  top_token_list = []
  for top_word_id in top_tokens_att[i]:
    word = tokenized_input[i][top_word_id.item()]
    # Remove the Ġ character from the token -- only applies for BART.
    word = word.replace("Ġ", "")
    top_token_list.append(word)

  return top_token_list


def print_model_output_stats(
    attention_analysis_obj,
    tokenizer,
    top_token_count=10,
    num_examples_to_print=5
    # MODEL_ALIAS, SAMPLE_LABEL, tokenized_input, output,
    # cls_att_mean, cls_att_grad_mean, top_token_count=10

):
  """Print stats for model output.

  Args:
    attention_analysis_obj (AttentionAnalysis): AttentionAnalysis object.
    tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
      Tokenizer object.
    top_token_count (int): Number of top tokens to print. Defaults to 10.
    num_examples_to_print (int): Number of examples to print. Defaults to 5.

  Returns:
    Prints to stdout.
    Sample output:
      Model: distilbert
        batch_size = 4 | seq_length = 363 | num_heads = 12 | num_layers = 6
      Example 1:
        Top Attention Tokens : wants, kills, is, a, good, this, doesn, t
        Top Gradient Tokens  : also, wants, kills, some, really, this, doesn, t
        Worst Gradient Tokens: do, com, ##men, ##d, is, a, good, for
      Example 2:
        [...]
  """
  print()
  print(f'============ {attention_analysis_obj.model_alias} ============')

  # Trim the number of examples to print if there are fewer examples.
  num_examples_to_print = min(
    num_examples_to_print, len(attention_analysis_obj.labels)
  )

  top_tokens_att = attention_analysis_obj.topk_attentions[:, :top_token_count]
  top_tokens_att_grad = attention_analysis_obj.topk_attentions_grad[:, :top_token_count]
  worst_tokens_att_grad = attention_analysis_obj.bottomk_attentions_grad[:, :top_token_count]
  labels = attention_analysis_obj.labels
  predicted_labels = attention_analysis_obj.predicted_labels
  tokenized_input = attention_analysis_obj.tokenized_inputs
  tokenized_input_list = [tokenizer.convert_ids_to_tokens(
      ids) for ids in tokenized_input.tolist()]

  for i in range(num_examples_to_print):
    tokens_att = get_top_token_list(tokenized_input_list, top_tokens_att, i)
    high_tokens_att_grad = get_top_token_list(
        tokenized_input_list, top_tokens_att_grad, i)
    low_tokens_att_grad = get_top_token_list(
        tokenized_input_list, worst_tokens_att_grad, i)

    print(
        f"Example {i+1}: LABEL = {labels[i]} | PREDICTED = {predicted_labels[i]}")
    print(f'  Input: {tokenizer.decode(tokenized_input[i])[:100]} [...]')
    print(f"  Top Attention Tokens : {', '.join(tokens_att)}")
    print(f"  Top Gradient Tokens  : {', '.join(high_tokens_att_grad)}")
    print(f"  Worst Gradient Tokens: {', '.join(low_tokens_att_grad)}")

  print()

if __name__ == '__main__':
  np.random.seed(42)
  torch.manual_seed(42)
  set_seed(42)

  MODEL_ALIAS = 'distilbert'
  DATA_SIZE = 4

  # Load Data
  test_loader = prepare_data_loader(
    model_alias=MODEL_ALIAS,
    sample_size=DATA_SIZE,
    batch_size=16)
  tokenizer = AutoTokenizer.from_pretrained(
      MODEL_DICT[MODEL_ALIAS]['pretrained_model'])

  # Run Attention Analysis
  attn_obj = AttentionAnalysis(MODEL_ALIAS, test_loader)
  attn_obj.run()

  attn_obj.cls_attention_means
  attn_obj.tokenized_inputs

  # 1 Sample Output
  tokens_1d = tokenizer.convert_ids_to_tokens(attn_obj.tokenized_inputs[0])
  att_weights_1d = attn_obj.cls_attention_means[0]

  print_attention_weights(tokens_1d, att_weights_1d)
  attention_weights_to_df(tokens_1d, att_weights_1d)
  print_model_output_stats(attn_obj, tokenizer)