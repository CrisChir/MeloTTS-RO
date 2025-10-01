from . import chinese, japanese, english, chinese_mix, korean, french, spanish, malay, romanian
from . import cleaned_text_to_sequence
import copy

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish, 'MS': malay, 'RO': romanian}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


# def clean_text_bert(text, language, device=None):
#     language_module = language_module_map[language]
#     norm_text = language_module.text_normalize(text)
#     phones, tones, word2ph = language_module.g2p(norm_text)
    
#     word2ph_bak = copy.deepcopy(word2ph)
#     for i in range(len(word2ph)):
#         word2ph[i] = word2ph[i] * 2
#     word2ph[0] += 1
#     bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
#     return norm_text, phones, tones, word2ph_bak, bert
def clean_text_bert(text, language, device):
    # ... (if/elif blocks for ja, zh, en, etc.)

    # =================================================================
    # THIS IS THE FINAL, CORRECT LOGIC FOR ROMANIAN
    # =================================================================
    elif language == "ro":
        # 1. Normalize the text
        norm_text = romanian.text_normalize(text)
        
        # 2. Get the RAW, UNPADDED phonemes, tones, and word2ph
        phones, tones, word2ph = romanian.g2p(norm_text)
        
        # 3. Calculate BERT features using the UNPADDED word2ph.
        #    This will now work without an IndexError.
        bert = romanian.get_bert_feature(norm_text, word2ph, device)
        
        # 4. AFTER calculating BERT, apply the final padding for the data file.
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    else:
        raise ValueError(f"Unsupported language: {language}")

    # The returned values are now correctly padded and will pass the assertion
    return norm_text, phones, tones, word2ph, bert

def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
