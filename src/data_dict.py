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
    'gpt2': {
        'pretrained_model': 'microsoft/DialogRPT-updown',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
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
    'tinybert_sid': {
        'pretrained_model': 'smiller324/imdb_tinybert',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'bert_suraj': {
        'pretrained_model': 'sshourie/test_trainer',
        'attention_param_name': 'attentions',
        'grad_attention_param_name': 'attentions'
    },
    'bart_suraj_cross_enc': {
        'pretrained_model': 'sshourie/BART_small_IMDB',
        'attention_param_name': 'cross_attentions',
        'grad_attention_param_name': 'encoder_attentions'
    },
    'bart_suraj_cross_cross': {
        'pretrained_model': 'sshourie/BART_small_IMDB',
        'attention_param_name': 'cross_attentions',
        'grad_attention_param_name': 'cross_attentions'
    },
    'bart_suraj_dec_dec': {
        'pretrained_model': 'sshourie/BART_small_IMDB',
        'attention_param_name': 'decoder_attentions',
        'grad_attention_param_name': 'decoder_attentions'
    },
}
