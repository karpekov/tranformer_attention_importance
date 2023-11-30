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
    num_examples_to_print=5,
    print_to_stdout=True

):
  """Print stats for model output.

  Args:
    attention_analysis_obj (AttentionAnalysis): AttentionAnalysis object.
    tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
      Tokenizer object.
    top_token_count (int): Number of top tokens to print. Defaults to 10.
    num_examples_to_print (int): Number of examples to print. Defaults to 5.

  Returns:
    output_list (list): List of sample input texts and their top tokens.

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

  output_list = []

  for i in range(num_examples_to_print):
    tokens_att = get_top_token_list(tokenized_input_list, top_tokens_att, i)
    high_tokens_att_grad = get_top_token_list(
        tokenized_input_list, top_tokens_att_grad, i)
    low_tokens_att_grad = get_top_token_list(
        tokenized_input_list, worst_tokens_att_grad, i)

    if print_to_stdout:
      print(
        f"Example {i+1}: LABEL = {labels[i]} | PREDICTED = {predicted_labels[i]}")
      print(f'  Input: {tokenizer.decode(tokenized_input[i])[:100]} [...]')
      print(f"  Top Attention Tokens : {', '.join(tokens_att)}")
      print(f"  Top Gradient Tokens  : {', '.join(high_tokens_att_grad)}")
      print(f"  Worst Gradient Tokens: {', '.join(low_tokens_att_grad)}")

    output_dict = {}
    output_dict['example_id'] = i
    output_dict['label'] = labels[i].item()
    output_dict['predicted_label'] = predicted_labels[i].item()
    output_dict['text'] = tokenizer.decode(
      tokenized_input[i], skip_special_tokens=True)
    output_dict['top_attention_tokens'] = tokens_att
    output_dict['top_attention_tokens_with_id'] = list(
        zip(tokens_att, top_tokens_att[i].tolist()))
    output_dict['top_gradient_tokens'] = high_tokens_att_grad
    output_dict['top_grad_attention_tokens_with_id'] = list(
        zip(high_tokens_att_grad, top_tokens_att_grad[i].tolist()))
    output_dict['worst_gradient_tokens'] = low_tokens_att_grad
    output_dict['worst_grad_attention_tokens_with_id'] = list(
        zip(low_tokens_att_grad, worst_tokens_att_grad[i].tolist()))

    output_list.append(output_dict)

  return output_list

def get_token_embeddings(attention_analysis_obj, tokenizer, topk=5):

  # Extract relevant attributes from attention analysis object.
  tokenized_inputs = attention_analysis_obj.tokenized_inputs
  labels = attention_analysis_obj.labels
  predicted_labels = attention_analysis_obj.predicted_labels
  hidden_states = attention_analysis_obj.hidden_states
  topk_attentions = attention_analysis_obj.topk_attentions[:, :topk]
  topk_attentions_grad = attention_analysis_obj.topk_attentions_grad[:, :topk]

  # Prep tensors to extract top-k tokens and their embeddings.
  data_size = tokenized_inputs.shape[0]
  hidden_dim_size = hidden_states.shape[2]
  top_vectors_att = torch.zeros(data_size, topk, hidden_dim_size)
  top_vectors_att_grad = torch.zeros(data_size, topk, hidden_dim_size)
  top_tokens_att = []
  top_tokens_att_grad = []

  # Loop through each example in the data and extract top-k tokens and embeds.
  for i in range(data_size):
    # Get hidden state vector for top-k tokens based on attention.
    indices_att = topk_attentions[i]
    top_vectors_att[i] = hidden_states[i, indices_att, :]
    tokens_att = tokenizer.convert_ids_to_tokens(
        tokenized_inputs[i, indices_att])
    top_tokens_att.append(tokens_att)
    # Get hidden state vector for top-k gradient tokens.
    indices_att_grad = topk_attentions_grad[i]
    top_vectors_att_grad[i] = hidden_states[i, indices_att_grad, :]
    tokens_att_grad = tokenizer.convert_ids_to_tokens(
        tokenized_inputs[i, indices_att_grad])
    top_tokens_att_grad.append(tokens_att_grad)

  # Prep final outputs for later embedding analysis.
  repeated_labels = torch.repeat_interleave(labels, topk)
  # Get flattened token list.
  top_tokens_att = [token for sublist in top_tokens_att for token in sublist]
  top_tokens_att_grad = [token for sublist in top_tokens_att_grad for token in sublist]
  # Flatten and stack embeddings tensors.
  top_vectors_att = top_vectors_att.reshape(-1, hidden_dim_size)
  top_vectors_att_grad = top_vectors_att_grad.reshape(-1, hidden_dim_size)

  return repeated_labels, top_tokens_att, top_tokens_att_grad, top_vectors_att, top_vectors_att_grad


if __name__ == '__main__':
  np.random.seed(42)
  torch.manual_seed(42)
  set_seed(42)

  MODEL_ALIAS = 'distilbert'
  DATA_SIZE = 32

  # Load Data
  test_loader = prepare_data_loader(
    model_alias=MODEL_ALIAS,
    sample_size=DATA_SIZE,
    batch_size=16)
  tokenizer = AutoTokenizer.from_pretrained(
      MODEL_DICT[MODEL_ALIAS]['pretrained_model'])

  # Run Attention Analysis
  attn_obj = AttentionAnalysis(
    MODEL_ALIAS, test_loader, output_hidden_states=True)
  attn_obj.run(store_input_output=True)

  attn_obj.cls_attention_means
  attn_obj.tokenized_inputs

  # Get top tokens and embeddings
  repeated_labels, top_tokens_att, top_tokens_att_grad, top_vectors_att, top_vectors_att_grad = get_token_embeddings(
      attn_obj, tokenizer)
  print(top_vectors_att.shape)

  # # 1 Sample Output
  # tokens_1d = tokenizer.convert_ids_to_tokens(attn_obj.tokenized_inputs[0])
  # att_weights_1d = attn_obj.cls_attention_means[0]

  # print_attention_weights(tokens_1d, att_weights_1d)
  # attention_weights_to_df(tokens_1d, att_weights_1d)
  # example_list = print_model_output_stats(
  #   attn_obj, tokenizer, top_token_count=20,
  #   num_examples_to_print=32, print_to_stdout=False)

  # # print(example_list)
  # example_df = pd.DataFrame(example_list)
  # example_df.to_csv('data/sample_examples_with_top_attention_tokens.csv', index=False)
