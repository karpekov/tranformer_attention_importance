"""
Script to load experiments data and visualize results.
"""

import pandas as pd
import numpy as np

import altair as alt
from plotnine import *
# from kinda_tidy import *
import kinda_tidy


def alt_horizontal_line(
    y_value: float = 0.0,
    y: str = 'y',
    color: str = 'black',
    size: int = 2,
    **line_kwargs
) -> alt.Chart:
  """Utility to plot a horizontal line in Altair.

  Args:
    y_value: Value where the horizontal line intersects y-axis.
    y: Name of the column (default should be fine for most cases).
    color: Color of the line, can be a string accepted by Vega lite or HEX.
    size: Width of the line.
    **line_kwargs: Additional keyword arguments passed to mark_rule().

  Returns:
    Altair Chart object with the horizontal line. Can be used within tidyverse
    package like this:
      df.alt_chart().mark(...).encode(...) + alt_horizontal_line().
  """
  temp_data = pd.DataFrame({y: [y_value]})
  line = alt.Chart(temp_data).mark_rule(
      color=color, size=size, **line_kwargs).encode(y=y)
  return line


def apply_common_configs(chart):
  """Apply common configs to Altair Chart object."""
  chart = (chart
    .properties(width=500, height=400)
    .configure_title(fontSize=16)
    .configure_axis(labelFontSize=12, titleFontSize=14)
    .configure_legend(titleFontSize=16, labelFontSize=14)
  )
  return chart


def plot_random_token_drop(
    df,
    title_annotation='Chart 1',
    metrics=['accuracy', 'f1'],
    model_name='BERT',
    bucket_size=10,
    to_apply_common_configs=True,
    save_chart=True,
    resolution_ppi=300,
    save_filename='random_tokens_drop'
):
  """Plot results of dropping random tokens.

  Args:
    df: DataFrame with results of dropping random tokens.
      Expected columns: tokens_dropped, 'accuracy', 'f1'
    title_annotation: Chart number.
    metrics: Metrics to plot, one of: ['accuracy', 'f1', 'precision', 'recall']
    model_name: Name of the model used to generate the results.
    bucket_size: Bucket size to aggregate results.
    to_apply_common_configs: Whether to apply common configs to the chart.
    save_chart: Whether to save the chart to a file.
    resolution_ppi: Resolution of the saved chart.

  Returns:
    If save_chart is False, returns:
      Altair Chart object with the plot.
    Else:
      Saves the chart to a file.
  """

  # Compute mean of metrics for each bucket, convert table from wide to long
  # format so that we can use `color` to plot different metrics on tjhe same
  # chart.
  temp_df = (df
      .assign(
          tokens_dropped=lambda df: (
              (df.tokens_dropped / bucket_size)).astype('int') * bucket_size
      )
      .groupby(['tokens_dropped'])
      [metrics]
      .agg('mean')
      .flatten_columns()
      .reset_index()

      .melt(
          id_vars='tokens_dropped',
          value_vars=metrics,
          var_name='metric',
          value_name='score'
      )
    )
  # Create chart object.
  chart = (temp_df
    .alt_chart(
        title=(
            f'{title_annotation} | {model_name.capitalize()}: '
          f'Dropping first {temp_df.tokens_dropped.max()} random tokens'
        )
    )
    .mark_line(size=2, point=True)
    .encode(
        x=alt.X('tokens_dropped', title='Number of Tokens Dropped'),
        y=alt.Y('score', axis=alt.Axis(format='%'), title='Metric Value'),
        color='metric'
    )
    + alt_horizontal_line(y_value=1.0, size=1, strokeDash=[6, 6])
  )
  # Apply common configs to the chart.
  if to_apply_common_configs:
    chart = apply_common_configs(chart)
  if save_chart:
    chart_num = '_'.join(title_annotation.lower().split(' '))
    save_filename = f'{chart_num}_{save_filename}_{model_name}'
    chart.save(f'charts/{save_filename}.png', ppi=resolution_ppi)
  else:
    return chart

def plot_all_drop_strategies_for_one_model(
    df,
    title_annotation='Chart 1',
    metric_name='accuracy',
    model_name='distilbert',
    to_apply_common_configs=True,
    save_chart=True,
    resolution_ppi=300,
    save_filename='drop_strategy_single_model'
):
  """Plot results of dropping tokens using various strtaegies for one model.

  Args:
    df: DataFrame with results of dropping tokens using various strategies.
      Expected columns: tokens_dropped, drop_strategy, model, accuracy
    title_annotation: Chart number, e.g.: "Chart 1".
    metric_name: Metric to plot, one of: ['accuracy', 'f1', 'precision', 'recall']
    model_name: Name of the model used to generate the results.
    to_apply_common_configs: Whether to apply common configs to the chart.
    save_chart: Whether to save the chart to a file.
    resolution_ppi: Resolution of the saved chart.

  Returns:
    If save_chart is False, returns:
      Altair Chart object with the plot.
    Else:
      Saves the chart to a file.
  """

  # Pre-process data.
  temp_df = (df
    .query(f'model == "{model_name}"')
    # Compute percentage of metric value relative to the first value.
    .sort_values(by='tokens_dropped')
    .groupby(['model', 'drop_strategy'], group_keys=False)
    .assign(
        perc_metric=lambda df: df[metric_name] /
        df[metric_name].iloc[0]
    )
  )

  chart = (temp_df
    .alt_chart(
        title=(
            f'{title_annotation} | {model_name.capitalize()}: '
            f'Faithfulness Test using Various Dropout Strategies'
        )
    )
    .mark_line(size=2, point=True)
    .encode(
        x=alt.X('tokens_dropped', title='Number of Tokens Dropped'),
        y=alt.Y(
          'perc_metric',
          title=f'% of baseline {metric_name}',
          axis=alt.Axis(format='%')),
        color='drop_strategy'
    )
    + alt_horizontal_line(y_value=1.0, size=1, strokeDash=[6, 6])
  )
  # Apply common configs to the chart.
  if to_apply_common_configs:
    chart = apply_common_configs(chart)
  if save_chart:
    chart_num = '_'.join(title_annotation.lower().split(' '))
    save_filename = f'{chart_num}_{save_filename}_{model_name}'
    chart.save(f'charts/{save_filename}.png', ppi=resolution_ppi)
  else:
    return chart

def plot_all_drop_strategies_for_all_models(
    df,
    title='Comparison of Dropout Strategies for Multiple Models',
    title_annotation='Chart 1',
    metric_name='accuracy',
    model_list=['distilbert'],
    drop_strategy_list=['attention', 'attention_grad_top'],
    color_scheme='tableau10',
    to_apply_common_configs=False,
    save_chart=True,
    resolution_ppi=300,
    save_filename='drop_strategy_all_models'
):
  """Plot results of dropping tokens using various strtaegies for all models.

  """
  # Pre-process data.
  temp_df = (df
    .query(f'model in {model_list} and drop_strategy in {drop_strategy_list}')
    # Compute percentage of metric value relative to the first value.
    .sort_values(by='tokens_dropped')
    .groupby(['model', 'drop_strategy'], group_keys=False)
    .assign(
        perc_metric=lambda df: df[metric_name] /
          df[metric_name].iloc[0]
    )
  )

  chart = ((temp_df
    .alt_chart(
        title=title,
        height=400,
        width=500
    )
    .mark_line(size=2, point=True)
    .encode(
        x=alt.X('tokens_dropped', title='Number of Tokens Dropped'),
        y=alt.Y('perc_metric', axis=alt.Axis(format='%'), title=f'% of baseline {metric_name}'),
        color=alt.Color(
            'model',
            scale=alt.Scale(scheme=color_scheme),
            legend=alt.Legend(
                titleFontSize=16, labelFontSize=16,
                symbolStrokeWidth=3,
                orient='none', legendX=554, legendY=270,
                fillColor='#E0E0E0', strokeColor='black',
                cornerRadius=5, padding=10
            )
        ),
        column=alt.Column(
            'drop_strategy', title='',
            header=alt.Header(labelFontSize=15)
        )
    )
    )
    .configure_title(fontSize=20, offset=20, anchor='middle')
    .configure_axis(labelFontSize=10, titleFontSize=14)
  )

  # Apply common configs to the chart.
  if to_apply_common_configs:
    chart = apply_common_configs(chart)
  if save_chart:
    chart_num = '_'.join(title_annotation.lower().split(' '))
    save_filename = f'{chart_num}_{save_filename}'
    chart.save(f'charts/{save_filename}.png', ppi=resolution_ppi)
  else:
    return chart

if __name__ == '__main__':

  # Load data.
  df_random = pd.read_csv(
      f'data/experiment_results_random_tokens_20231130_0329.csv')
  output_df = pd.read_csv('data/experiment_results_ALL.csv')

  # Create plots.
  plot_random_token_drop(df_random, 'Chart 1')
  plot_all_drop_strategies_for_one_model(output_df, 'Chart 2')
  plot_all_drop_strategies_for_all_models(
    output_df,
    title='BERT & BART: Attention vs Grad Attention',
    title_annotation='Chart 4',
    model_list=['distilbert', 'tinybert', 'roberta', 'bart_enc_attn'],
    color_scheme='dark2'
  )
  plot_all_drop_strategies_for_all_models(
      output_df,
      title='BART: Comparing Attention Mechanisms: \
        Encoder Attn, Decoder Attn, Cross Attn',
      title_annotation='Chart 3',
      model_list=['bart_enc_attn', 'bart_cross_attn', 'bart_dec_attn'],
      color_scheme='set1'
  )
