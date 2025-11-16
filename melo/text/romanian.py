import os, epitran
from functools import cache
from transformers import AutoTokenizer
from num2words import num2words

def distribute_phone(phone_len, word_len):
    if word_len == 0: return []
    phones_per_word = phone_len // word_len
    remaining_phones = phone_len % word_len
    phone_distribution = []
    for i in range(word_len):
        phones = phones_per_word
        if i < remaining_phones: phones += 1
        phone_distribution.append(phones)
    return phone_distribution

# model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'
model_id = 'readerbench/RoBERT-large'

@cache
def get_tokenizer(): return AutoTokenizer.from_pretrained(model_id)
@cache
def get_phonemizer_epitran(): return epitran.Epitran('ron-Latn')
def text_normalize(text):
    words = [num2words(word, lang='ro') if word.isdigit() else word for word in text.split()]
    text = ' '.join(words)
    if text and text[-1] not in ".?!": text += "."
    return text

def g2p(text):
    epi, tokenizer = get_phonemizer_epitran(), get_tokenizer()
    tokenized = tokenizer.tokenize(text)
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"): ph_groups.append([t])
        else: ph_groups[-1].append(t.replace("#", ""))
    phones, tones, word2ph = [], [], []
    for group in ph_groups:
        word = "".join(group)
        phonemized_word = epi.transliterate(word)
        phone_list = list(phonemized_word)
        for p in phone_list:
            phones.append(p)
            tones.append(1 if p == 'ˈ' else 0)
        word2ph += distribute_phone(len(phone_list), len(group))
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from . import romanian_bert
    return romanian_bert.get_bert_feature(text, word2ph, device=device)

# def get_bert_feature(text, word2ph, device=None, phone_len=None):
#     model, tokenizer = get_model_and_tokenizer()
#     if device:
#         model = model.to(device)

#     inputs = tokenizer(text, return_tensors="pt")
#     for i in inputs:
#         inputs[i] = inputs[i].to(model.device)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         last_hidden_state = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

#     # Exclude [CLS] and [SEP]
#     hidden = last_hidden_state[1:-1]  # [num_tokens, hidden_dim]
#     num_tokens = hidden.shape[0]

#     # Safeguard: truncate word2ph if too long
#     if len(word2ph) > num_tokens:
#         print(f"[WARN] word2ph too long ({len(word2ph)} > {num_tokens}), truncating.")
#         word2ph = word2ph[:num_tokens]

#     # Build phoneme-aligned embeddings
#     bert_features = []
#     for i, ph_count in enumerate(word2ph):
#         token_embedding = hidden[i]
#         bert_features.extend([token_embedding] * ph_count)

#     if not bert_features:
#         return torch.empty(model.config.hidden_size, 0).to(model.device)

#     bert_tensor = torch.stack(bert_features).T  # [hidden_dim, num_phoneme]

#     # Optional alignment
#     if phone_len is not None:
#         if bert_tensor.shape[1] > phone_len:
#             bert_tensor = bert_tensor[:, :phone_len]
#         elif bert_tensor.shape[1] < phone_len:
#             pad = torch.zeros(model.config.hidden_size, phone_len - bert_tensor.shape[1], device=model.device)
#             bert_tensor = torch.cat([bert_tensor, pad], dim=1)

#     # Final check
#     print(f"[INFO] BERT tensor shape: {bert_tensor.shape}, expected phoneme count: {sum(word2ph)}")
#     return bert_tensor


print("✅ SUCCESS: Self-contained romanian.py created.")
