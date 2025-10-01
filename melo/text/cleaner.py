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
    if language == "ja":
        norm_text = japanese.text_normalize(text)
        phones, tones, word2ph = japanese.g2p(norm_text)
        bert = japanese.get_bert_feature(norm_text, word2ph, device)
    elif language == "zh":
        norm_text = chinese.text_normalize(text)
        phones, tones, word2ph = chinese.g2p(norm_text)
        bert = chinese.get_bert_feature(norm_text, word2ph, device)
    elif language == "en":
        norm_text = english.text_normalize(text)
        phones, tones, word2ph = english.g2p(norm_text)
        bert = english.get_bert_feature(norm_text, word2ph, device)
    elif language == "ko":
        norm_text, phones, tones, word2ph = korean.clean_text(text)
        bert = korean.get_bert_feature(norm_text, word2ph, device)
    elif language == "fr":
        norm_text = french.text_normalize(text)
        phones, tones, word2ph = french.g2p(norm_text)
        bert = french.get_bert_feature(norm_text, word2ph, device)
    elif language == "ms":
        norm_text = malay.text_normalize(text)
        phones, tones, word2ph = malay.g2p(norm_text)
        bert = malay.get_bert_feature(norm_text, word2ph, device)

    # This is the new, correctly structured block for Romanian
    elif language == "ro":
        # 1. Normalize the text
        norm_text = romanian.text_normalize(text)
        
        # 2. Get the RAW, UNPADDED phonemes, tones, and word2ph
        phones, tones, word2ph = romanian.g2p(norm_text)
        
        # 3. Calculate BERT features using the UNPADDED word2ph
        bert = romanian.get_bert_feature(norm_text, word2ph, device)
        
        # 4. AFTER calculating BERT, apply the final padding for the data file
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    
    else:
        raise ValueError(f"Unsupported language: {language}")

    # This final return statement applies to all languages
    return norm_text, phones, tones, word2ph, bert

def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
