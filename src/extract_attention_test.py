import unittest
from parameterized import parameterized_class

from extract_attention import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_processing import *
from data_dict import MODEL_DICT

from extract_attention import *


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
# @parameterized_class(('model_alias',), [(model_alias,) for model_alias in ['distilbert', 'bert', 'roberta_sid', 'bert_souraj']])
class TestAttentionAnalysis(unittest.TestCase):
  model_alias = None

  @classmethod
  def setUpClass(cls):
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

    cls.model_dict = MODEL_DICT

  def test_run_produces_non_empty_output(self):
    # Initialize the AttentionAnalysis class with the mock DataLoader
    analysis = AttentionAnalysis(
        model_alias=self.model_alias,
        data_loader=self.data_loader,
        model_dict=self.model_dict
    )

    # Run the analysis
    analysis.run(top_k=5)

    # Assert that the results are non-empty
    self.assertIsNotNone(analysis.cls_attention_means)
    self.assertIsNotNone(analysis.cls_attention_grad_means)
    self.assertIsNotNone(analysis.topk_attentions)
    self.assertIsNotNone(analysis.topk_attentions_grad)
    self.assertIsNotNone(analysis.bottomk_attentions_grad)
    self.assertIsNotNone(analysis.tokenized_inputs)
    self.assertIsNotNone(analysis.labels)
    self.assertIsNotNone(analysis.predicted_labels)

    # Additionally, assert that results are of expected shape
    self.assertEqual(analysis.cls_attention_means.shape[0], 4)
    self.assertEqual(analysis.cls_attention_grad_means.shape[0], 4)
    self.assertEqual(analysis.topk_attentions.shape[0], 4)
    self.assertEqual(analysis.topk_attentions.shape[1], 5)
    self.assertEqual(analysis.topk_attentions_grad.shape[0], 4)
    self.assertEqual(analysis.topk_attentions_grad.shape[1], 5)
    self.assertEqual(analysis.bottomk_attentions_grad.shape[0], 4)
    self.assertEqual(analysis.bottomk_attentions_grad.shape[1], 5)
    self.assertEqual(analysis.tokenized_inputs.shape[0], 4)
    self.assertEqual(analysis.tokenized_inputs.shape[1], 512)
    self.assertEqual(analysis.labels.shape[0], 4)
    self.assertEqual(analysis.predicted_labels.shape[0], 4)

if __name__ == '__main__':
    unittest.main()