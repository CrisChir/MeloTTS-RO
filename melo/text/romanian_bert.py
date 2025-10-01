# melo/text/romanian_bert.py

import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer

@cache
def get_model_and_tokenizer():
    model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, tokenizer

def get_bert_feature(text, word2ph, device=None):
    model, tokenizer = get_model_and_tokenizer()
    if device:
        model = model.to(device)

    inputs = tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)

    # Align BERT features with phonemes
    # The tokenizer might split words into sub-tokens. word2ph maps phoneme counts to these sub-tokens.
    bert_features = []
    # We skip the [CLS] and [SEP] tokens by slicing [1:-1]
    for i, ph_count in enumerate(word2ph):
        # The index i corresponds to the i-th sub-token
        token_embedding = last_hidden_state[i + 1]
        # Repeat the sub-token embedding for each phoneme it corresponds to
        bert_features.extend([token_embedding] * ph_count)

    return torch.stack(bert_features)
