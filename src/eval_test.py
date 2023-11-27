import unittest
import torch
from eval import SentimentAnalysisPipeline
from eval import prepare_data_loader
from utils import *

# Core tests of the SentimentAnalysisPipeline class using distilbert.
class TestSentimentAnalysisPipeline_distilbert(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='distilbert')

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

# Evaluation-only tests of BERT model.
class TestSentimentAnalysisPipeline_bert(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='bert')

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

# Evaluation-only tests of BART model.
class TestSentimentAnalysisPipeline_bart(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='bart')

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

# Evaluation-only tests of BART model.


class TestSentimentAnalysisPipeline_roberta_sid(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='roberta_sid')

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


class TestSentimentAnalysisPipeline_bert_souraj(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='bert_souraj')

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


class TestSentimentAnalysisPipeline_bart_souraj(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load a very small sample of the input dataset once for all tests
    cls.data_loader = prepare_data_loader(sample_size=10, batch_size=8)

    # Initialize the SentimentAnalysisPipeline instance
    cls.pipeline = SentimentAnalysisPipeline(model_alias='bart_souraj')

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

if __name__ == '__main__':
  unittest.main()
