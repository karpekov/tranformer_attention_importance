from transformers import pipeline, set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from datasets import load_dataset, DatasetDict

import accelerate

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pprint import pprint

MODEL_DICT = {
    # ================== HUGGINGFACE MODELS ==================
    'bert': {
        'pretrained_model': 'yoshitomo-matsubara/bert-large-uncased-sst2',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'distilbert': {
        'pretrained_model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'bart': {
        'pretrained_model': 'valhalla/bart-large-sst2',
        # 'pretrained_model': 'facebook/bart-large-mnli',
        # All BART attentions: ['decoder_attentions', 'cross_attentions', 'encoder_attentions']
        # Decoder attention is BAD. Use either encoder or cross.
        # 'attention_param_name': 'decoder_attentions'
        # 'attention_param_name': 'cross_attentions'
        'attention_param_name': 'cross_attentions',
        'grad_attention_param_name': 'encoder_attentions'
    },
    # 'gpt2': {
    #     'pretrained_model': 'microsoft/DialogRPT-updown',
    #     'attention_param_name': 'attentions',
    #     'grad_attention_param_name': 'attentions'
    # },
    # 'llama2': {
    #   'pretrained_model': 'meta-llama/Llama-2-7b',
    #   'attention_param_name': 'attentions'
    # },
    # ================ PERSONAL FINETUNED MODELS ================
    'roberta_sid': {
        'pretrained_model': 'smiller324/imdb_roberta',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'bert_souraj': {
        'pretrained_model': 'sshourie/test_trainer',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'bart_souraj': {
        'pretrained_model': 'sshourie/BART_small_IMDB',
        'attention_param_name': 'cross_attentions',
        'grad_attention_param_name': 'encoder_attentions'
    },
}

def get_data(name='imdb', sample_perc=None):
  """Load full or sample dataset.

  Args:
  - name (str): Dataset name. By default loads the IMDB dataset.
  - sample_perc (float): Percent of the total dataset to load.
                 By default loads the full dataset.

  Returns:
  - DatasetDict of form:
      DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 25000 * sample_perc
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 25000 * sample_perc
        })
    })

  """
  if not sample_perc:
    output_dataset = load_dataset(name)
  else:
    train_dataset = load_dataset(
        name, split='train')
    test_dataset = load_dataset(name, split='test')

    train_size = train_dataset.num_rows
    train_sample_size = int(train_size * sample_perc)
    test_size = test_dataset.num_rows
    test_sample_size = int(test_size * sample_perc)

    train_sample_indices = np.random.choice(
        train_size, size=train_sample_size, replace=False)
    test_sample_indices = np.random.choice(
        test_size, size=test_sample_size, replace=False)
    output_dataset = DatasetDict({
      'train': train_dataset.select(train_sample_indices),
      'test': test_dataset.select(test_sample_indices)
    })

  return output_dataset


def get_CLS_attention_per_token(model_name, input_text, labels, return_cls_only=False):
  """ Compute average last layer attention from each token to [CLS] token.

  Args:
    model_name: Name of HuggingFace model. Must be in MODEL_DICT.
    input (List[str] | str): Text to compute attention weights over.
    labels (List[int] | int): Label(s) for input text.

  Returns:
    input (transformers.tokenization_utils_base.BatchEncoding):
      Tokenized input. Can access these tensors as follows:
        - input['input_ids']
        - input['attention_mask']
      For each input, can also extract tokens like this:
        - input[0].tokens
    output (transformers.modeling_outputs.SequenceClassifierOutput):
      Model output. Contains logits and attention weights.
      Can access these tensors as follows:
        - output.logits
        - output.attentions
    cls_attention_mean (torch.Tensor):
      2D Tensor of attention weights.
      Shape: (batch_size, sequence_length).
    cls_attention_grad_mean (torch.Tensor):
      2D Tensor of attention weights gradients.
      Shape: (batch_size, sequence_length).
    NOTE: Will return cls_attention[_grad]_mean if return_cls_only is True.
  """
  # If input and labels are just for one example, convert them to lists.
  if not isinstance(input_text, list):
    test = [input_text]
  if not isinstance(labels, list):
    labels = [labels]

  # Convert labels to tensor.
  label_tensor = torch.tensor(labels, dtype=torch.long)

  # Extract model-specific info from model dict:
  pretrained_model_name = MODEL_DICT[model_name]['pretrained_model']
  attention_name = MODEL_DICT[model_name]['attention_param_name']
  grad_attention_name = MODEL_DICT[model_name]['grad_attention_param_name']

  # Instantiate the model:
  model = AutoModelForSequenceClassification.from_pretrained(
      pretrained_model_name,
      output_attentions=True,
      num_labels=2,
      ignore_mismatched_sizes=True
  )
  # Pre-process the input text, make sure it's padded to the lenght of the
  # longest input string and truncated at the model's max input len
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
  # Pre-process the input text, make sure it's padded to the lenght of the
  # longest input string and truncated at the model's max input lenght.
  tokenized_input = tokenizer(input_text, return_tensors="pt",
                    padding=True, truncation=True)
  # Run inference.
  output = model(**tokenized_input)

  # Do a backward pass to get gradients.
  for attention in output[attention_name]:
    attention.retain_grad()
  for attention in output[grad_attention_name]:
    attention.retain_grad()
  loss_function = nn.CrossEntropyLoss()
  loss = loss_function(output.logits, label_tensor)
  loss.backward(retain_graph=True)

  # Grab all attention heads from the last layer. Shape:
  #   (batch_size, num_heads, sequence_length, sequence_length).
  attention = output[attention_name][-1]
  # Get the dL/da gradient for each attention head. Shape:
  #   (batch_size, num_heads, sequence_length, sequence_length).
  attention_grad = output[grad_attention_name][-1].grad
  # Grab attention weights for the CLS token only (index=0). Shape:
  #   (batch_size, num_heads, sequence_length)
  cls_attention = attention[:, :, 0, :]
  cls_attention_grad = attention_grad[:, :, 0, :]
  # Compute average of CLS attention across all attention heads. Output shape:
  #   (batch_size, sequence_length)
  # (TODO:alexkarpekov): Potentially compute MAX attention weigths, or
  #                      keep attention from all heads.
  cls_attention_mean = cls_attention.mean(dim=1)
  # Following the paper, compute the gradient-weighted average of the attention:
  #   -1 * (dL/da * a)
  # Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9671639
  cls_attention_grad_mean = -(cls_attention_grad * cls_attention).mean(dim=1)
  if return_cls_only:
    return cls_attention_mean, cls_attention_grad_mean
  else:
    return tokenized_input, output, cls_attention_mean, cls_attention_grad_mean

def _random_swap(attention_row, num_bits):
  """Helper function to swap random bits in an attention mask row to 0."""
  num_cols = attention_row.size(0)
  rand_indices = torch.randperm(num_cols)[:num_bits]
  attention_row[rand_indices] = 0


def _swap_with_indices(attention_row, swap_indices, num_bits):
  """Helper function to swap specified bits in an attention mask row to 0."""
  # Only use the first num_bits indices
  valid_indices = swap_indices[:num_bits]
  # Filter out any indices that are out of bounds
  valid_indices = valid_indices[valid_indices < attention_row.size(0)]
  attention_row[valid_indices] = 0


def disable_attention_mask_bits(
    attention_tensor,
    num_bits=10,
    swap_index_tensor=None
):
  """Set specified bits in the attention mask to '0', or random bits if no indices are specified.

  Args:
    attention_tensor (torch.Tensor): 2D attention mask where each row is a mask.
    num_bits (int): Number of bits to set to '0' for each mask.
    swap_index_tensor (torch.Tensor): Optional 2D tensor with the same number of rows as attention_tensor,
                                      containing indices to swap to '0'.

  Returns:
    torch.Tensor: Modified 2D attention mask.

  Raises:
    ValueError: If swap_index_tensor has a different number of rows than attention_tensor.
  """
  # Clone the tensor to avoid modifying it in place
  attention_tensor_copy = attention_tensor.clone()

  if swap_index_tensor is not None:
    if swap_index_tensor.size(0) != attention_tensor_copy.size(0):
      raise ValueError(
        f"swap_index_tensor must have the same number of rows (current: {swap_index_tensor.size(0)}) as attention_tensor (input rows: {attention_tensor_copy.size(0)})")
    for row in range(attention_tensor_copy.size(0)):
      _swap_with_indices(
        attention_tensor_copy[row], swap_index_tensor[row], num_bits)
  else:
    for row in range(attention_tensor_copy.size(0)):
      _random_swap(attention_tensor_copy[row], num_bits)

  return attention_tensor_copy

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
    tokenized_input (transformers.tokenization_utils_base.BatchEncoding):
      Tokenized input. For each input, can also extract tokens like this:
        - input[0].tokens
    i (int): Index of example in batch.

  Returns:
    top_token_list (list): List of tokens.
  """
  top_token_list = []
  for (tok_id, tok) in enumerate(tokenized_input[i].tokens):
    if tok_id in top_tokens_att.indices[i]:
      # Remove the Ġ character from the token -- only applies for BART.
      tok = tok.replace("Ġ", "")
      top_token_list.append(tok)
  return top_token_list

def print_model_output_stats(
        MODEL_ALIAS, SAMPLE_LABEL, tokenized_input, output,
        cls_att_mean, cls_att_grad_mean, top_token_count=10
        ):
  """Print stats for model output.

  Args:
    All args are returned from get_CLS_attention_per_token().
    top_token_count (int): Number of top tokens per example to print.

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
  num_layers = len(output[MODEL_DICT[MODEL_ALIAS]['attention_param_name']])
  (batch_size, num_heads, seq_length, _
   ) = output[MODEL_DICT[MODEL_ALIAS]['attention_param_name']][-1].shape
  print(f"\nModel: {MODEL_ALIAS}")
  print(f"  {batch_size = } | {seq_length = } | {num_heads = } | {num_layers = }\n")

  top_tokens_att = cls_att_mean.topk(top_token_count)
  top_tokens_att_grad = cls_att_grad_mean.topk(top_token_count)
  worst_tokens_att_grad = cls_att_grad_mean.topk(top_token_count, largest=False)

  predicted_labels = output.logits.argmax(dim=1)

  for i in range(batch_size):
    tokens_att = get_top_token_list(tokenized_input, top_tokens_att, i)
    high_tokens_att_grad = get_top_token_list(tokenized_input, top_tokens_att_grad, i)
    low_tokens_att_grad = get_top_token_list(tokenized_input, worst_tokens_att_grad, i)

    print(f"Example {i+1}: LABEL = {SAMPLE_LABEL[i]} | PREDICTED = {predicted_labels[i]}")
    print(f"  Top Attention Tokens : {', '.join(tokens_att)}")
    print(f"  Top Gradient Tokens  : {', '.join(high_tokens_att_grad)}")
    print(f"  Worst Gradient Tokens: {', '.join(low_tokens_att_grad)}")

  print()

if __name__ == '__main__':

  np.random.seed(42)
  torch.manual_seed(42)
  set_seed(42)

  imdb = get_data("imdb", 0.05)

  # ==== ATTENTION WIEGHTS SWAPPING ====

  DATA_SIZE = 1
  SAMPLE_TEXT = imdb['test']['text'][:DATA_SIZE]
  SAMPLE_LABEL = imdb['test']['label'][:DATA_SIZE]

  # MODEL_ALIAS = 'yoshitomo-matsubara/bert-large-uncased-sst2'
  MODEL_ALIAS = 'distilbert'
  MODEL_NAME = MODEL_DICT[MODEL_ALIAS]['pretrained_model']

  # Instantiate the model:
  model = AutoModelForSequenceClassification.from_pretrained(
      MODEL_NAME,
      output_attentions=True,
      num_labels=2,
      ignore_mismatched_sizes=True
  )
  # Pre-process the input text, make sure it's padded to the lenght of the
  # longest input string and truncated at the model's max input len
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # Pre-process the input text, make sure it's padded to the lenght of the
  # longest input string and truncated at the model's max input lenght.
  tokenized_input = tokenizer(SAMPLE_TEXT, return_tensors="pt",
                              padding=True, truncation=True)
  # Run inference.
  output = model(**tokenized_input)

  new_mask = disable_attention_mask_bits(
    tokenized_input.attention_mask,
    num_bits=50
    )
  print(new_mask)

  cls_att_mean, cls_att_grad_mean = get_CLS_attention_per_token(
      MODEL_ALIAS,
      SAMPLE_TEXT,
      SAMPLE_LABEL,
      return_cls_only=True
  )

  cls_att_mean_top_indices = cls_att_mean.topk(10).indices

  new_mask2 = disable_attention_mask_bits(
    tokenized_input.attention_mask,
    num_bits=200,
    swap_index_tensor=cls_att_mean_top_indices
    )
  print(new_mask2)

  print()
