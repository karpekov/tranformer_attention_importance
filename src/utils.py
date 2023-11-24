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
    'bert': {
        'pretrained_model': 'yoshitomo-matsubara/bert-large-uncased-sst2',
        'attention_param_name': 'attentions'
    },
    'distilbert': {
        'pretrained_model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'attention_param_name': 'attentions'
    },
    'bart': {
        'pretrained_model': 'valhalla/bart-large-sst2',
        # All BART attentions: ['decoder_attentions', 'cross_attentions', 'encoder_attentions']
        # Decoder attention is BAD. Use either encoder or cross.
        # 'attention_param_name': 'decoder_attentions'
        'attention_param_name': 'cross_attentions'
        # 'attention_param_name': 'encoder_attentions'
    },
    'gpt2': {
        'pretrained_model': 'microsoft/DialogRPT-updown',
        'attention_param_name': 'attentions'
    },
    # 'llama2': {
    #   'pretrained_model': 'meta-llama/Llama-2-7b',
    #   'attention_param_name': 'attentions'
    # },
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

# def preprocess_function(examples):
#   """Tokenize text."""
#   return tokenizer(examples['text'], truncation=True, padding=True)


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

  # Instantiate the model:
  model = AutoModelForSequenceClassification.from_pretrained(
      pretrained_model_name, output_attentions=True)
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
  loss_function = nn.CrossEntropyLoss()
  loss = loss_function(output.logits, label_tensor)
  loss.backward(retain_graph=True)

  # Grab all attention heads from the last layer. Shape:
  #   (batch_size, num_heads, sequence_length, sequence_length).
  attention = output[attention_name][-1]
  # Get the dL/da gradient for each attention head. Shape:
  #   (batch_size, num_heads, sequence_length, sequence_length).
  attention_grad = output[attention_name][-1].grad
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
    .style
    # Red, green
    .bar(subset=['attention'], align='mid', color=['#d65f5f', '#5fba7d'])
    )
  return df


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
    tokens_att = [tok for (tok_id, tok) in enumerate(
        tokenized_input[i].tokens) if tok_id in top_tokens_att.indices[i]]
    high_tokens_att_grad = [tok for (tok_id, tok) in enumerate(
        tokenized_input[i].tokens) if tok_id in top_tokens_att_grad.indices[i]]
    low_tokens_att_grad = [tok for (tok_id, tok) in enumerate(
        tokenized_input[i].tokens) if tok_id in worst_tokens_att_grad.indices[i]]
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

  # ==== TEST ONE MODEL ====
  MODEL_ALIAS = 'distilbert'
  SAMPLE_TEXT = imdb['train']['text'][:4]
  SAMPLE_LABEL = imdb['train']['label'][:4]

  (tokenized_input, output,
   cls_att_mean, cls_att_grad_mean) = get_CLS_attention_per_token(
     MODEL_ALIAS,
     SAMPLE_TEXT,
     SAMPLE_LABEL,
     return_cls_only=False
  )

  print(cls_att_mean.shape)
  print(cls_att_grad_mean.shape)

  PRINT_INDEX = 0
  print_attention_weights(
      tokenized_input[PRINT_INDEX].tokens, cls_att_mean[PRINT_INDEX])

  print()
  # ==== TEST ALL MODELS ====
  SAMPLE_TEXT = imdb['train']['text'][:4]
  SAMPLE_LABEL = imdb['train']['label'][:4]
  for MODEL_ALIAS in MODEL_DICT.keys():

    (tokenized_input, output,
     cls_att_mean, cls_att_grad_mean) = get_CLS_attention_per_token(
      MODEL_ALIAS,
      SAMPLE_TEXT,
      SAMPLE_LABEL,
      return_cls_only=False
    )

    print_model_output_stats(
        MODEL_ALIAS, SAMPLE_LABEL, tokenized_input, output, cls_att_mean, cls_att_grad_mean)

    print()





