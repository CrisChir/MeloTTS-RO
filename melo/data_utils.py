import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_filepaths_and_text
from utils import load_wav_to_torch_librosa as load_wav_to_torch
from text import cleaned_text_to_sequence, get_bert
import numpy as np

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_sid_text, hparams):
        self.hparams = hparams
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.spk_map = hparams.spk2id
        self.disable_bert = getattr(hparams, "disable_bert", False)
        self.use_mel_spec_posterior = getattr(hparams, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        audiopaths_sid_text_new, lengths = [], []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(self.audiopaths_sid_text):
            try: _id, spk, language, text, phones, tone, word2ph = item
            except: print(item); raise
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append([audiopath, spk, language, text, phones, tone, word2ph])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else: skipped += 1
        if lengths: logger.info(f'min: {min(lengths)}; max: {max(lengths)}')
        logger.info(f"skipped: {skipped}, total: {len(self.audiopaths_sid_text)}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
        bert, ja_bert, phones, tone, language = self.get_text(text, word2ph, phones, tone, language, audiopath)
        spec, wav = self.get_audio(audiopath)
        sid = int(self.spk_map.get(sid, 0))
        sid = torch.LongTensor([sid])
        return (phones, spec, wav, sid, tone, language, bert, ja_bert)

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio_norm.unsqueeze(0)
        if self.use_mel_spec_posterior:
            spec = mel_spectrogram_torch(audio_norm, self.filter_length, self.n_mel_channels, self.sampling_rate, self.hop_length, self.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax, center=False)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)): word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        
        INPUT_DATA_PREFIX = '/kaggle/input/'
        OUTPUT_BERT_PREFIX = '/kaggle/working/bert_features/'
        relative_path = os.path.relpath(wav_path, INPUT_DATA_PREFIX)
        bert_path = os.path.join(OUTPUT_BERT_PREFIX, relative_path.replace(".wav", ".bert.pt"))
        
        try:
            bert = torch.load(bert_path)
            assert bert.shape[-1] == len(phone)
        except Exception as e:
            # --- THIS IS THE FIX ---
            # We add the missing 'device' argument to the get_bert call.
            # We will default to 'cpu' as this happens in a dataloader worker.
            print(f"Error loading BERT file: {bert_path}, attempting to regenerate on CPU.", e)
            bert = get_bert(text, word2ph, language_str, 'cpu') 
            torch.save(bert, bert_path)
            assert bert.shape[-1] == len(phone), phone
            # --- END OF FIX ---

        if self.disable_bert:
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(768, len(phone))
        else:
            if language_str in ["ZH"]:
                bert = bert
                ja_bert = torch.zeros(768, len(phone))
            elif language_str in ["JP", "EN", "ZH_MIX_EN", "KR", 'SP', 'ES', 'FR', 'DE', 'RU', 'MS', 'RO']:
                ja_bert = bert
                bert = torch.zeros(1024, len(phone))
            else:
                raise ValueError(f"Unsupported language for BERT handling: {language_str}")

        assert bert.shape[-1] == len(phone)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, phone, tone, language

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

# The rest of the file is standard and has no errors
class TextAudioSpeakerCollate:
    # ... (omitted for brevity, it is correct)

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    # ... (omitted for brevity, it is correct)
