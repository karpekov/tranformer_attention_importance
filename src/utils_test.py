import unittest
from utils import *
from datasets import load_dataset

imdb = get_data("imdb", 0.05)


class UtilsTest(unittest.TestCase):
  def test_CLS_attention_for_single_model(self):
    """Test CLS attention per token function for a single model."""
    model_alias = 'distilbert'
    data_size = 4
    sample_text = imdb['test']['text'][:data_size]
    sample_label = imdb['test']['label'][:data_size]

    tokenized_input, output, cls_att_mean, cls_att_grad_mean = get_CLS_attention_per_token(
      model_alias,
      sample_text,
      sample_label,
      return_cls_only=False
    )

    self.assertEqual(cls_att_mean.shape[0], data_size)
    self.assertEqual(cls_att_grad_mean.shape[0], data_size)

    print_index = 0
    print_attention_weights(
      tokenized_input[print_index].tokens, cls_att_mean[print_index])

    print_model_output_stats(
      model_alias, sample_label, tokenized_input, output, cls_att_mean, cls_att_grad_mean)

  def test_CLS_attention_for_multiple_models(self):
    """Test CLS attention per token function for multiple models."""
    data_size = 4
    sample_text = imdb['test']['text'][:data_size]
    sample_label = imdb['test']['label'][:data_size]

    for model_alias in MODEL_DICT.keys():
      tokenized_input, output, cls_att_mean, cls_att_grad_mean = get_CLS_attention_per_token(
        model_alias,
        sample_text,
        sample_label,
        return_cls_only=False
      )

      self.assertEqual(cls_att_mean.shape[0], data_size)
      self.assertEqual(cls_att_grad_mean.shape[0], data_size)

      print_model_output_stats(
        model_alias, sample_label, tokenized_input, output, cls_att_mean, cls_att_grad_mean)


if __name__ == '__main__':
  unittest.main()
