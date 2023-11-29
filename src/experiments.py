import os
import pandas as pd
from datetime import datetime
import torch

from data_dict import MODEL_DICT
from eval import SentimentAnalysisPipeline
from data_processing import prepare_data_loader
from extract_attention import AttentionAnalysis

class ModelEvaluationExperiment:
  def __init__(self, model_dict, data_loader, max_token_drop):
    self.model_dict = model_dict
    self.data_loader = data_loader
    self.max_token_drop = max_token_drop
    self.results = []

  def run(
      self,
      run_random=True,
      run_attention=True,
      run_grad_attention_top=True,
      run_grad_attention_bottom=True,
      top_k=50,
      store_input_output=False
  ):
    """ Run all experiments on all models in the model_dict.

    Args:
      run_random (bool): Run the random token drop experiment.
      run_attention (bool): Run the attention-based token drop experiment.
      run_grad_attention_top (bool):
        Run the gradient-based token drop experiment.
      run_grad_attention_bottom (bool):
        Run the gradient-based token drop experiment.
    """
    # Check that at least one experiment was specified.
    if not(run_random or run_attention or run_grad_attention_top or run_grad_attention_bottom):
      raise ValueError('No Experiments were specified.')
    for model_alias in self.model_dict.keys():
      print(f'=========================================================')
      print(f'===============| RUNNING EVALS for {model_alias}')
      print(f'---------------------------------------------------------')
      # Run the model and extract attention weights and gradients.
      if run_attention or run_grad_attention_top or run_grad_attention_bottom:
        try:
          print(f'===============| Running AttentionAnalysis() |===========')
          attention_analysis_obj = AttentionAnalysis(
              model_alias,
              self.data_loader,
              model_dict=self.model_dict,
          )
          attention_analysis_obj.run(
              top_k=top_k, store_input_output=store_input_output)
        except Exception as e:
          print(f'Failed to run attention analysis for {model_alias}')
          print(f"An exception occurred: {e}")
          # Skip current model and jump to the next in the loop.
          continue

      # Run multiple token drop strategies and evaluate each model separately.
      for tokens_to_drop in range(self.max_token_drop + 1):
        print(f'=== tokens_to_drop: {tokens_to_drop}')
        if run_random:
          self._run_sentiment_analysis_pipe(
              model_alias, tokens_to_drop, 'random')
        if run_attention:
          self._run_sentiment_analysis_pipe(
              model_alias,
              tokens_to_drop,
              drop_strategy_name='attention',
              swap_index_tensor=attention_analysis_obj.topk_attentions
          )
        if run_grad_attention_top:
          self._run_sentiment_analysis_pipe(
              model_alias,
              tokens_to_drop,
              drop_strategy_name='attention_grad_top',
              swap_index_tensor=attention_analysis_obj.topk_attentions_grad
          )
        if run_grad_attention_bottom:
          self._run_sentiment_analysis_pipe(
              model_alias,
              tokens_to_drop,
              drop_strategy_name='attention_grad_bottom',
              swap_index_tensor=attention_analysis_obj.bottomk_attentions_grad,
          )
      # Save results after each model. Note that this will overwrite the file.
      self.save_results()
      # Excplicitely delete the attention analysis object to free up memory.
      if run_attention or run_grad_attention_top or run_grad_attention_bottom:
        del attention_analysis_obj
      torch.cuda.empty_cache()

  def _run_sentiment_analysis_pipe(
      self,
      model_alias,
      tokens_to_drop,
      drop_strategy_name='no_name',
      swap_index_tensor=None
  ):
    """Run the sentiment analysis pipeline on a model."""
    try:
      pipeline = SentimentAnalysisPipeline(
          model_alias=model_alias,
          model_dict=self.model_dict
      )
      metrics = pipeline.evaluate_model(
          self.data_loader,
          num_bits_to_flip=tokens_to_drop,
          swap_index_tensor=swap_index_tensor
      )
      self.results.append({
          'model': model_alias,
          'drop_strategy': drop_strategy_name,
          'input_selection': 'all',
          'tokens_dropped': tokens_to_drop,
          'accuracy': metrics['accuracy'],
          'precision': metrics['precision'],
          'recall': metrics['recall'],
          'f1': metrics['f1'],
          'status': 'success'
      })
    # If an error occurs, store the error message in the results.
    except Exception as e:
      self.results.append({
          'model': model_alias,
          'drop_strategy': drop_strategy_name,
          'input_selection': 'all',
          'tokens_dropped': tokens_to_drop,
          'accuracy': None,
          'precision': None,
          'recall': None,
          'f1': None,
          'status': 'error',
          'error_message': str(e)
      })

  def save_results(
      self,
      filename='data/experiment_results',
      append_timestamp=True
  ):
    """Save the results to a csv file."""
    results_df = pd.DataFrame(self.results)
    if not os.path.exists('data'):
      os.makedirs('data')
    if append_timestamp:
      current_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
      filename = f'{filename}_{current_timestamp}'
    results_df.to_csv(f'{filename}.csv', index=False)

  def get_results(self):
    return pd.DataFrame(self.results)

if __name__ == '__main__':

  MODEL_DICT = {
    # 'bert': {
    #     'pretrained_model': 'yoshitomo-matsubara/bert-large-uncased-sst2',
    #     'attention_param_name': 'attentions',
    #     'grad_attention_param_name': 'attentions'
    # },
    # 'distilbert': {
    #     'pretrained_model': 'distilbert-base-uncased-finetuned-sst-2-english',
    #     'attention_param_name': 'attentions',
    #     'grad_attention_param_name': 'attentions'
    # 'tinybert_sid': {
    #     'pretrained_model': 'smiller324/imdb_tinybert',
    #     'attention_param_name': 'attentions',
    #     'grad_attention_param_name': 'attentions'
    # },
    'gpt2': {
        'pretrained_model': 'microsoft/DialogRPT-updown',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
  }

  DATA_SIZE = 4

  test_loader = prepare_data_loader(
    model_alias=list(MODEL_DICT.keys())[0],
    sample_size=DATA_SIZE,
    batch_size=16
  )

  experiment = ModelEvaluationExperiment(
      MODEL_DICT, test_loader, max_token_drop=4)
  experiment.run(top_k=30)
  experiment.save_results('data/experiment_results', append_timestamp=False)
