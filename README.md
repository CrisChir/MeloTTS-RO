<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

Original README at https://github.com/myshell-ai/MeloTTS

## Introduction
MeloTTS RO is a forked of https://github.com/mesolitica/MeloTTS-MS based on https://github.com/myshell-ai/MeloTTS to support Malay and hopefully Romanian language.

## Improvement

1. Use c and Malaya Speech normalizer, [melo/text/malay.py](melo/text/malay.py),add [melo/text/malay.py](melo/text/romanian.py) added rodiacritic to requirements

```python
text = 'hello nama saya.'
text = text_normalize(text)
phones, tones, word2ph = g2p(text)
"""
(['_',
'h',
'ˈɛ',
'l',
'o',
'n',
'ˈa',
'm',
'ə',
's',
'ˈa',
'j',
'ə',
'.',
'_'],
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 4, 4, 4, 1, 1])
"""
```


1.1 To do and test 1. Use `ro` phonemizer and Romanian Speech normalizer,


2. Use Pretrained Malaysian BERT, [melo/text/malay_bert.py](melo/text/malay_bert.py).
   2.2
   ```
    test a bert model for romanian language added [melo/text/romanian_bert.py](melo/text/romanian_bert.py)
   ```
   
4. Extend symbols, [melo/text/symbols.py](melo/text/symbols.py).
   
```
Extend romanian: MeloTTS-RO/melo/text/symbols.py

ro_symbols = [
"a",
"ə",
"b",
"d",
"dʒ",
"e",
"f",
"ɡ",
"h",
"i",
"ɨ",
"k",
"l",
"m",
"n",
"o",
"p",
"r",
"s",
"ʃ",
"t",
"ts",
"tʃ",
"u",
"v",
"z",
"ʒ",
"j",
"w",
"e̯a",
"o̯a",
"ă",
"â",
"î",
"ș",
"ț"
]
num_ro_tones = 1
```
6. Hardcode the size of vocab and tone based on pretrained but use the new size during inference, [melo/models.py](melo/models.py), 

```python
self.enc_p = TextEncoder(
    n_vocab if is_eval else 219,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    gin_channels=self.enc_gin_channels,
    num_languages=num_languages,
    num_tones=num_tones if is_eval else 16,
)
```

5. Use the official pretrained models after that extend the embedding size, [melo/train.py](melo/train.py),


```
python
utils.load_checkpoint(
  hps.pretrain_G,
  net_g,
  None,
  skip_optimizer=True
)

old_embeddings = net_g.module.enc_p.emb
net_g.module.enc_p.emb = net_g.module.get_resized_embeddings(old_embeddings, len(symbols))

old_embeddings = net_g.module.enc_p.tone_emb
net_g.module.enc_p.tone_emb = net_g.module.get_resized_embeddings(old_embeddings, 18)

print(net_g.module.enc_p.emb.weight.shape, net_g.module.enc_p.tone_emb.weight.shape)
```
6. added a new function to utils
```
def distribute_phone(phone_len, word_len):
    """
    Distributes the phoneme count across the word's sub-tokens.
    For example, if a word has 5 phonemes and 2 sub-tokens,
    it might distribute them as [3, 2].
    """
    phones_per_word = phone_len // word_len
    remaining_phones = phone_len % word_len
    phone_distribution = []
    for i in range(word_len):
        phones = phones_per_word
        if i < remaining_phones:
            phones += 1
        phone_distribution.append(phones)
    return phone_distribution
```
   

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
