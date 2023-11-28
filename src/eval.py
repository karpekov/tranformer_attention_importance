"""
Creates SentimentAnalysisPipeline class to run a model and evaluate it on data.
The pipeline can also flip bits in the attention mask, either randomly or
based on a tensor of indices.

See usage in '__main__' at the bottom of this file.
"""

import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from extract_attention import *
from data_processing import *
from data_dict import MODEL_DICT

class SentimentAnalysisPipeline:
  def __init__(self, model_alias='distilbert', model_dict=MODEL_DICT, device=None):
    """Initializes the pipeline

    Args:
      model_alias (str, optional): Alias of the model to use. Defaults to 'distilbert'.
      model_dict (dict, optional):
        Dictionary containing the model aliases and pretrained models.
        Should be defined in the utils module. Defaults to MODEL_DICT.
      device ([type], optional): Device to use for training. Defaults to None.
    """
    self.model_name = model_dict[model_alias]['pretrained_model']
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        self.model_name)
    self.device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

  def evaluate_model(
      self,
      data_loader,
      num_bits_to_flip=None,
      swap_index_tensor=None
  ):
    """Evaluates the model on data. Optionally flips bits in the attention mask.

    If only num_bits_to_flip is specified, random bits will be flipped.
    If only swap_index_tensor is specified, all indices in the tensor will be
      flipped.
    If both num_bits_to_flip and swap_index_tensor are specified, only the
      first num_bits_to_flip indices in each row of the swap_index_tensor
      will be flipped.

    Args:
      data_loader ([torch.utils.data.DataLoader]): Dataloader to evaluate on.
      num_bits_to_flip (int, optional): Number of bits to flip in the
        attention mask. Defaults to None.
      swap_index_tensor (torch.Tensor, optional): 2D tensor containing
        indices to swap to '0'. Defaults to None.

    Returns:
      dict: Evaluation metrics (accuracy, precision, recall, f1)
    """
    self.model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
      for i, batch in enumerate(tqdm(data_loader, desc="Evaluation")):
        labels = batch.pop('label').cpu().numpy()
        # Flip the attention mask bits for the current batch, if specified.
        if num_bits_to_flip is not None or swap_index_tensor is not None:
          if swap_index_tensor is not None:
            # Calculate the start and end indices for the current batch.
            start_idx = i * data_loader.batch_size
            end_idx = min(start_idx + data_loader.batch_size,
                          len(swap_index_tensor))
            current_swap_indices = swap_index_tensor[start_idx:end_idx]
          else:
            current_swap_indices = None
          batch['attention_mask'] = self._disable_attention_mask_bits(
            batch['attention_mask'],
            num_bits=num_bits_to_flip,
            swap_index_tensor=current_swap_indices
          )
        # Continue evaluation as normal.
        batch = self._to_device(batch)
        outputs = self.model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
        all_true_labels.extend(labels)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
      all_true_labels,
      all_predictions,
      average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

  def _encode_batch(self, batch):
    """Encodes a batch of data using the model tokenizer

    Args:
      batch (dict): Batch of data to encode

    Returns:
      dict: Encoded batch
    """
    return self.tokenizer(
      batch['text'],
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )


  def _to_device(self, batch):
    """Moves a batch to the device"""
    return {k: v.to(self.device) for k, v in batch.items()}

  def _random_swap(self, attention_row, num_bits):
    """Helper function to swap random bits in an attention mask row to 0."""
    num_cols = attention_row.size(0)
    rand_indices = torch.randperm(num_cols)[:num_bits]
    attention_row[rand_indices] = 0

  def _swap_with_indices(self, attention_row, swap_indices, num_bits):
    """Helper function to swap specified bits in an attention mask row to 0."""
    # Only use the first num_bits indices
    valid_indices = swap_indices[:num_bits]
    # Filter out any indices that are out of bounds
    valid_indices = valid_indices[valid_indices < attention_row.size(0)]
    attention_row[valid_indices] = 0

  def _check_swap_index_tensor(self, swap_index_tensor, attention_tensor):
    """Check if swap_index_tensor has the same num of rows as attention_tensor."""
    if swap_index_tensor.size(0) != attention_tensor.size(0):
      raise ValueError(f"swap_index_tensor must have the same number of rows "
        f"(current: {swap_index_tensor.size(0)}) as attention_tensor "
        f"(input rows: {attention_tensor.size(0)})")

  def _disable_attention_mask_bits(
      self,
      attention_tensor,
      num_bits=10,
      swap_index_tensor=None
  ):
    """Set specified bits in the attention mask to '0', or random bits
      if no indices are specified.

    Args:
      attention_tensor (torch.Tensor): 2D attention mask where each row is a mask.
      num_bits (int): Number of bits to set to '0' for each mask.
      swap_index_tensor (torch.Tensor): Optional 2D tensor with the same number of
        rows as attention_tensor, containing indices to swap to '0'.

    Returns:
      torch.Tensor: Modified 2D attention mask.

    Raises:
      ValueError: If swap_index_tensor has a different number of rows
        than attention_tensor.
    """
    # Clone the tensor to avoid modifying it in place
    attention_tensor_copy = attention_tensor.clone()

    if swap_index_tensor is not None:
      self._check_swap_index_tensor(swap_index_tensor, attention_tensor_copy)
      for row in range(attention_tensor_copy.size(0)):
        self._swap_with_indices(
            attention_tensor_copy[row], swap_index_tensor[row], num_bits)
    else:
      for row in range(attention_tensor_copy.size(0)):
        self._random_swap(attention_tensor_copy[row], num_bits)

    return attention_tensor_copy


if __name__ == '__main__':
  # Usage
  # Replace with your model name if different
  model_name = 'distilbert'
  DATA_SIZE = 50

  pipeline = SentimentAnalysisPipeline(model_alias='distilbert')
  test_loader = prepare_data_loader(sample_size=DATA_SIZE, batch_size=16)

  # Mask 50 bits from a tensor of indices.
  # Create a tensor of indices to swap to '0' of size DATA_SIZE x 50:
  # DATA_SIZE: number of samples in the test set
  # Only flip top 10 bits in each row.
  swap_index_tensor = torch.randint(0, 100, size=(DATA_SIZE, 50))
  test_scores = pipeline.evaluate_model(
      test_loader, num_bits_to_flip=10, swap_index_tensor=swap_index_tensor)
  print(test_scores)
