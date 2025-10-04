import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger

# --- THIS IS THE FIX ---
# We add the missing imports that were causing the NameError and other issues.
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
import utils
from text import cleaned_text_to_sequence, get_bert
# --- END OF FIX ---

"""Multi speaker version"""

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        # This is the correct __init__ function that expects the FULL hps object
        self.hparams = hparams
        self.audiopaths_sid_text = utils.load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = self.hparams.data.max_wav_value
        self.sampling_rate = self.hparams.data.sampling_rate
        self.filter_length = self.hparams.data.filter_length
        self.hop_length = self.hparams.data.hop_length
        self.win_length = self.hparams.data.win_length
        self.spk_map = self.hparams.data.spk2id

        # Correctly access model and data parameters from the full hps object
        self.disable_bert = getattr(self.hparams.model, "disable_bert", False)
        self.use_mel_spec_posterior = getattr(self.hparams.model, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(self.hparams.data, "n_mel_channels", 80)
        
        self.cleaned_text = getattr(self.hparams.data, "cleaned_text", False)
        self.add_blank = self.hparams.data.add_blank
        self.min_text_len = getattr(self.hparams.data, "min_text_len", 1)
        self.max_text_len = getattr(self.hparams.data, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        audiopaths_sid_text_new, lengths = [], []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(self.audiopaths_sid_text):
            try:
                _id, spk, language, text, phones, tone, word2ph = item
            except:
                print(item); raise
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones.split(" ")) and len(phones.split(" ")) <= self.max_text_len:
                phones_list = phones.split(" ")
                tone_list = [int(i) for i in tone.split(" ")]
                word2ph_list = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append([audiopath, spk, language, text, phones_list, tone_list, word2ph_list])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}' )
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
        audio_norm, sampling_rate = utils.load_wav_to_torch_librosa(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio_norm.unsqueeze(0)
        if self.use_mel_spec_posterior:
            spec = mel_spectrogram_torch(audio_norm, self.filter_length, self.n_mel_channels, self.sampling_rate, self.hop_length, self.win_length, self.hparams.data.mel_fmin, self.hparams.data.mel_fmax, center=False)
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
            print(f"Error loading BERT file: {bert_path}, attempting to regenerate.", e)
            bert = get_bert(text, word2ph, language_str)
            torch.save(bert, bert_path)
            assert bert.shape[-1] == len(phone), phone

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
    def __init__(self, return_ids=False): self.return_ids = return_ids
    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
        max_text_len = max([len(x[0]) for x in batch]); max_spec_len = max([x[1].size(1) for x in batch]); max_wav_len = max([x[2].size(1) for x in batch])
        text_lengths = torch.LongTensor(len(batch)); spec_lengths = torch.LongTensor(len(batch)); wav_lengths = torch.LongTensor(len(batch)); sid = torch.LongTensor(len(batch))
        text_padded, tone_padded, language_padded = torch.LongTensor(len(batch), max_text_len), torch.LongTensor(len(batch), max_text_len), torch.LongTensor(len(batch), max_text_len)
        bert_padded, ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len), torch.FloatTensor(len(batch), 768, max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_(); tone_padded.zero_(); language_padded.zero_(); spec_padded.zero_(); wav_padded.zero_(); bert_padded.zero_(); ja_bert_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text, spec, wav, sid_val, tone, language, bert, ja_bert = row
            text_padded[i, : text.size(0)] = text; text_lengths[i] = text.size(0)
            spec_padded[i, :, : spec.size(1)] = spec; spec_lengths[i] = spec.size(1)
            wav_padded[i, :, : wav.size(1)] = wav; wav_lengths[i] = wav.size(1)
            sid[i] = sid_val
            tone_padded[i, : tone.size(0)] = tone
            language_padded[i, : language.size(0)] = language
            bert_padded[i, :, : bert.size(1)] = bert
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, tone_padded, language_padded, bert_padded, ja_bert_padded

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths; self.batch_size = batch_size; self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        print('buckets:', self.num_samples_per_bucket)
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]; idx_bucket = self._bisect(length)
            if idx_bucket != -1: buckets[idx_bucket].append(i)
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0: buckets.pop(i); self.boundaries.pop(i + 1)
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i]); total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets: indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets: indices.append(list(range(len(bucket))))
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]; len_bucket = len(bucket)
            if len_bucket == 0: continue
            ids_bucket = indices[i]; num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
    def _bisect(self, x, lo=0, hi=None):
        if hi is None: hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]: return mid
            elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
            else: return self._bisect(x, mid + 1, hi)
        else: return -1
    def __len__(self): return self.num_samples // self.batch_size
