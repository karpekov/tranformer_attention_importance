# Project Description
[TBA]

# Key Findings
[TBA]

![DistilBERT dropout strategies comparison](charts/chart_2_drop_strategy_single_model_distilbert.png)

![BART attention mechanism comparison](charts/chart_3_drop_strategy_all_models.png)

![All models side by side comparison](charts/chart_4_drop_strategy_all_models.png)

# How to run experiments
[TBA]

# Code structure
[TBA]

# Environment setup
To create a working environemnt using conda:
```
conda env create -f dlt_project_env.yml
```

To activate the environment:
```
conda activate dlt_project_env
```

To update the environment in case you added new dependencies:
```
conda env update -f dlt_project_env.yml
```

Update the `requirements.txt` file:
```
pip list --format=freeze > requirements.txt
```