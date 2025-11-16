import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from melo.text.cleaner import clean_text_bert
import os
import torch
from melo.text.symbols import symbols, num_languages, num_tones

@click.command()
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    base_dir = os.path.dirname(metadata)

    if train_path is None:
        train_path = os.path.join(base_dir, 'train.list')
    if val_path is None:
        val_path = os.path.join(base_dir, 'val.list')
    out_config_path = os.path.join(base_dir, 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        print("🔧 Starting preprocessing...")
        with open(cleaned_path, 'w', encoding='utf-8') as out_file:
            with open(metadata, 'r', encoding='utf-8') as in_file:
                for line in tqdm(in_file.readlines()):
                    try:
                        utt, spk, language, text = line.strip().split("|", 3)

                        norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda')

                        assert len(phones) == len(tones)
                        assert len(phones) == sum(word2ph)

                        out_file.write(
                            "{}|{}|{}|{}|{}|{}|{}\n".format(
                                utt, spk, language, norm_text,
                                " ".join(phones),
                                " ".join([str(i) for i in tones]),
                                " ".join([str(i) for i in word2ph]),
                            )
                        )

                        # 🔄 Save .bert.pt locally in same folder as .wav
                        bert_path = utt.replace(".wav", ".bert.pt")
                        output_bert_path = os.path.join(base_dir, "wavs", os.path.basename(bert_path))
                        os.makedirs(os.path.dirname(output_bert_path), exist_ok=True)
                        torch.save(bert.cpu().T, output_bert_path)

                    except Exception as error:
                        print(f"❌ Error processing line: {line.strip()}", repr(error))

        metadata = cleaned_path

    print("📂 Splitting into train and validation lists...")
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list, val_list = [], []
    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f: f.writelines(train_list)
    with open(val_path, "w", encoding="utf-8") as f: f.writelines(val_list)

    print("🛠 Updating configuration file...")
    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    main()
