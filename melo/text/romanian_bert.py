import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer

@cache
def get_model_and_tokenizer():
    """
    Loads and caches the Romanian BERT model and tokenizer.
    This function is called only once.
    """
    # model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'
    model_id = 'readerbench/RoBERT-large'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, tokenizer

def get_bert_feature(text, word2ph, device=None):
    """
    Extracts BERT features for the given text and aligns them with phonemes.
    - text: The normalized text input.
    - word2ph: A list where each element is the number of phonemes for a corresponding BERT sub-token.
    - device: The computing device ('cuda' or 'cpu').
    """
    model, tokenizer = get_model_and_tokenizer()
    if device:
        model = model.to(device)

    inputs = tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)
    
    bert_features = []
    # The number of tokens from BERT tokenizer should match the length of word2ph.
    # We subtract 2 to account for the special [CLS] and [SEP] tokens.
    num_bert_tokens = last_hidden_state.shape[0] - 2
    if len(word2ph) != num_bert_tokens:
        # This is a safeguard against rare tokenization mismatches.
        # It truncates the word2ph list to prevent an IndexError.
        word2ph = word2ph[:num_bert_tokens]
        
    for i, ph_count in enumerate(word2ph):
        # We get the embedding for the i-th token, skipping the [CLS] token at index 0.
        token_embedding = last_hidden_state[i + 1]
        
        # Repeat the token's embedding for each phoneme it corresponds to.
        bert_features.extend([token_embedding] * ph_count)
    
    if not bert_features:
        # Handle cases where no features are generated (e.g., empty input).
        return torch.empty(0, model.config.hidden_size)

    # Stack the list of tensors into a single tensor.
    return torch.stack(bert_features)
