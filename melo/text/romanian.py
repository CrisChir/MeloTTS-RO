
import os
from functools import cache
from transformers import AutoTokenizer
from num2words import num2words
from melo.text.utils import distribute_phone
import epitran

model_id = os.environ.get('MODEL_ID', 'dumitrescustefan/bert-base-romanian-cased-v1')

@cache
def get_tokenizer():
    return AutoTokenizer.from_pretrained(model_id)

class RomanianNormalizer:
    def normalize(self, text, add_fullstop=True):
        words = []
        for word in text.split():
            if word.isdigit():
                words.append(num2words(int(word), lang='ro'))
            else:
                words.append(word)
        text = ' '.join(words)
        if add_fullstop and text[-1] not in ".?!":
            text += "."
        return text

@cache
def get_normalizer():
    return RomanianNormalizer()

@cache
def get_phonemizer_epitran():
    # Use 'ron-Latn' for Romanian in Epitran
    return epitran.Epitran('ron-Latn')

def text_normalize(text):
    return get_normalizer().normalize(text)

def g2p(text):
    """
    This function now uses Epitran for Grapheme-to-Phoneme conversion.
    It returns raw, unpadded lists, as required by the cleaner.
    """
    epi = get_phonemizer_epitran()
    tokenizer = get_tokenizer()
    tokenized = tokenizer.tokenize(text)
    
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    phones, tones, word2ph = [], [], []
    for group in ph_groups:
        word = "".join(group)
        # Use Epitran's transliterate method to get IPA
        phonemized_word = epi.transliterate(word)
        
        # Epitran returns a single string of IPA characters.
        # We can treat each character as a phoneme.
        phone_list = list(phonemized_word)
        
        for p in phone_list:
            phones.append(p)
            # Use the IPA standard stress mark 'ˈ' to determine tones
            tones.append(1 if p == 'ˈ' else 0)
            
        word2ph += distribute_phone(len(phone_list), len(group))
        
    return phones, tones, word2ph

# The get_bert_feature function remains the same
def get_bert_feature(text, word2ph, device=None):
    from melo.text import romanian_bert
    return romanian_bert.get_bert_feature(text, word2ph, device=device)
