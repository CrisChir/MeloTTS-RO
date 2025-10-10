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

print("✅ SUCCESS: Self-contained romanian.py created.")
