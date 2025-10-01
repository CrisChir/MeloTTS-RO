# melo/text/romanian.py

import os
from functools import cache
from transformers import AutoTokenizer
from num2words import num2words
# from ro_diacritics import restore_diacritics # be aware fasttext is not supported 

# Import from our new utility file
from .utils import distribute_phone

# Set the MODEL_ID to the Romanian BERT model
#It is just a poposed approach not tested from here https://github.com/dumitrescustefan/Romanian-Transformers?tab=readme-ov-file
model_id = os.environ.get('MODEL_ID', 'dumitrescustefan/bert-base-romanian-cased-v1')

@cache
def get_tokenizer():
    """Loads the Romanian BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

class RomanianNormalizer:
    """A simple normalizer for Romanian text."""
    def normalize(self, text, add_fullstop=True):
        # Convert numbers to words in Romanian
        words = []
        for word in text.split():
            if word.isdigit():
                words.append(num2words(int(word), lang='ro'))
            else:
                words.append(word)
        text = ' '.join(words)
        #### there is an incompatibility with torchtext
        # # Restore diacritics
        # text = restore_diacritics(text)
        ######

        # Ensure the text ends with punctuation
        if add_fullstop and text[-1] not in ".?!":
            text += "."
            
        return text

@cache
def get_normalizer():
    """Returns an instance of our custom Romanian normalizer."""
    return RomanianNormalizer()

@cache
def get_phonemizer():
    """Initializes the phonemizer for Romanian."""
    import phonemizer
    from phonemizer.separator import Separator

    # Use 'ro' for Romanian language
    backend = phonemizer.backend.EspeakBackend(language='ro', preserve_punctuation=True, with_stress=True)
    separator = Separator(phone='-', word='|')
    return backend, separator

def text_normalize(text):
    """Normalizes the input text using the Romanian normalizer."""
    normalizer = get_normalizer()
    normalized_text = normalizer.normalize(text, add_fullstop=True)
    return normalized_text

# def g2p(text, pad_start_end=True):
#     """Converts graphemes to phonemes for Romanian text."""
#     global_phonemizer, separator = get_phonemizer()
#     tokenizer = get_tokenizer()
    
#     # Tokenize text into sub-tokens (e.g., 'bună ziua' -> ['bună', 'ziua'])
#     tokenized = tokenizer.tokenize(text)
    
#     # Group sub-tokens back into whole words
#     # Example: ['recuno', '##ștință'] -> [['recuno', 'ștință']]
#     ph_groups = []
#     for t in tokenized:
#         if not t.startswith("#"):
#             ph_groups.append([t])
#         else:
#             ph_groups[-1].append(t.replace("#", ""))

#     phones = []
#     tones = []
#     word2ph = []
    
#     for group in ph_groups:
#         word = "".join(group)
#         word_len = len(group) # Number of sub-tokens in this word

#         # Phonemize the complete word
#         phonemized_word = global_phonemizer.phonemize([word], separator=separator)[0].replace('|', '').strip()
        
#         # Split into individual phonemes
#         splitted = [p for p in phonemized_word.split('-') if p]
#         phone_len = len(splitted)
        
#         for s in splitted:
#             phones.append(s)
#             # Assign tone based on stress marker
#             tones.append(1 if 'ˈ' in s else 0)

#         # Distribute the total phone count among the sub-tokens of the word
#         word2ph += distribute_phone(phone_len, word_len)

#     if pad_start_end:
#         phones = ["_"] + phones + ["_"]
#         tones = [0] + tones + [0]
#         # word2ph doesn't need padding here as it maps to BERT tokens, which have CLS/SEP
        
#     return phones, tones, word2ph

def g2p(text, pad_start_end=True):
    """Converts graphemes to phonemes for Romanian text."""
    global_phonemizer, separator = get_phonemizer()
    tokenizer = get_tokenizer()
    tokenized = tokenizer.tokenize(text)
    
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    phones = []
    tones = []
    word2ph = []
    
    for group in ph_groups:
        word = "".join(group)
        word_len = len(group)
        phonemized_word = global_phonemizer.phonemize([word], separator=separator)[0].replace('|', '').strip()
        splitted = [p for p in phonemized_word.split('-') if p]
        phone_len = len(splitted)
        
        for s in splitted:
            phones.append(s)
            tones.append(1 if 'ˈ' in s else 0)

        word2ph += distribute_phone(phone_len, word_len)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        # =================================================================
        # ADD THIS LINE BACK - THIS IS THE FIX
        # It adds a count of 1 for the start token and 1 for the end token.
        word2ph = [1] + word2ph + [1]
        # =================================================================
        
    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    """Imports and uses the Romanian BERT feature extractor."""
    try:
        # Assuming the new file is in the same directory
        from . import romanian_bert
    except ImportError:
        # Fallback for running as a script
        import romanian_bert

    return romanian_bert.get_bert_feature(text, word2ph, device=device)

# --- Example Usage ---
if __name__ == "__main__":
    # Test with a Romanian sentence
    text = 'Bună ziua, numele meu este 123'
    
    # 1. Normalize the text
    normalized_text = text_normalize(text)
    print(f"Normalized Text: '{normalized_text}'")
    
    # 2. Convert to phonemes
    phones, tones, word2ph = g2p(normalized_text)
    
    print("\n--- G2P Output ---")
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2Ph:", word2ph)

    # 3. Get BERT features (requires torch and a GPU/CPU)
    print("\n--- BERT Feature Extraction ---")
    # We need to re-tokenize for the BERT model input, as g2p's tokenization is internal
    bert_tokenizer = get_tokenizer()
    bert_tokens = bert_tokenizer.tokenize(normalized_text)
    print(f"BERT Tokens ({len(bert_tokens)}): {bert_tokens}")
    print(f"Word2Ph Mapping ({len(word2ph)}): {word2ph}")

    # Ensure the lengths match for the feature extractor
    if len(bert_tokens) == len(word2ph):
        features = get_bert_feature(normalized_text, word2ph)
        print(f"BERT feature shape: {features.shape}")
        # Expected shape: [Number of Phonemes, BERT Hidden Size (e.g., 768)]
        print(f"Matches total phonemes? {features.shape[0] == len(phones) - 2}") # Subtract 2 for padding
    else:
        print("Mismatch between BERT tokens and word2ph mapping. Cannot extract features.")
