import unittest
from parameterized import parameterized_class

import torch
from torch.utils.data import Dataset

from eval import SentimentAnalysisPipeline
from eval import prepare_data_loader
from extract_attention import *
from data_processing import *
from data_dict import MODEL_DICT


class NamedTensorDataset(Dataset):
  def __init__(self, input_ids, attention_mask, labels):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return {
        'input_ids': self.input_ids[idx],
        'attention_mask': self.attention_mask[idx],
        'label': self.labels[idx]
    }


@parameterized_class(('model_alias',), [(model_alias,) for model_alias in MODEL_DICT.keys()])
# @parameterized_class(('model_alias',), [(model_alias,) for model_alias in ['distilbert']])
class TestSentimentAnalysisPipeline(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # # Load a very small sample of the input dataset once for all tests
    # cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)
    # Mock the dataset
    cls.texts = [
        "Attention is all you need",
        "Deep Bidirectional Transformers",
        "Low-Rank Adaptation of Large Language Models",
        "Global Vectors for Word Representation"
    ]
    cls.labels = [0, 1, 0, 1]
    # cls.model_alias = 'distilbert'
    cls.model_name = MODEL_DICT[cls.model_alias]['pretrained_model']
    cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    # Tokenize the mock texts
    tokenized_batch = cls.tokenizer(
        cls.texts,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    mock_dataset = NamedTensorDataset(
        tokenized_batch['input_ids'],
        tokenized_batch['attention_mask'],
        torch.tensor(cls.labels)
    )
    cls.data_loader = DataLoader(mock_dataset, batch_size=2)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias=cls.model_alias)

  def test_evaluation_no_params(self):
    # Test the evaluation with default parameters
    metrics = self.pipeline.evaluate_model(self.data_loader)
    self.assertIsInstance(metrics, dict)
    self.assertTrue('accuracy' in metrics)

  def test_evaluation_random_indices(self):
    # Test the evaluation with random index replacement
    metrics = self.pipeline.evaluate_model(
        self.data_loader, num_bits_to_flip=5)
    self.assertIsInstance(metrics, dict)
    self.assertTrue('accuracy' in metrics)

  def test_evaluation_specific_indices(self):
    # Test the evaluation with a specific index matrix
    # Assuming the attention mask length is 512
    swap_index_tensor = torch.randint(0, 512, (10, 5))
    metrics = self.pipeline.evaluate_model(
        self.data_loader, swap_index_tensor=swap_index_tensor)
    self.assertIsInstance(metrics, dict)
    self.assertTrue('accuracy' in metrics)

  def test_evaluation_specific_indices_with_num_bits(self):
    # Test the evaluation with a specific index matrix
    # Assuming the attention mask length is 512
    swap_index_tensor = torch.randint(0, 512, (10, 5))
    metrics = self.pipeline.evaluate_model(
        self.data_loader,
        num_bits_to_flip=10,
        swap_index_tensor=swap_index_tensor
    )
    self.assertIsInstance(metrics, dict)
    self.assertTrue('accuracy' in metrics)

  def test_encode_batch(self):
    # Test the _encode_batch method
    batch = {'text': ["This is a test", "Another test"]}
    encoded_batch = self.pipeline._encode_batch(batch)
    self.assertTrue(
        'input_ids' in encoded_batch and 'attention_mask' in encoded_batch)

  def test_to_device(self):
    # Test the _to_device method
    batch = {'input_ids': torch.tensor(
        [[1, 2], [3, 4]]), 'attention_mask': torch.tensor([[1, 1], [1, 0]])}
    batch_to_device = self.pipeline._to_device(batch)
    self.assertTrue(
        all(tensor.device == self.pipeline.device for tensor in batch_to_device.values()))

if __name__ == '__main__':
  unittest.main()
