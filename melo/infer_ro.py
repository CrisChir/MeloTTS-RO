import os
import click
import numpy as np
from melo.api import TTS
from melo.utils import split_sentences
import soundfile as sf

@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file or directory")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
def main(ckpt_path, text, language, output_dir):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")

    # 🔧 Auto-select latest checkpoint if a directory is passed
    if os.path.isdir(ckpt_path):
        ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith("G_") and f.endswith(".pth")]
        if not ckpt_files:
            raise FileNotFoundError(f"No G_*.pth files found in {ckpt_path}")
        ckpt_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        ckpt_path = os.path.join(ckpt_path, ckpt_files[-1])
        print(f"✅ Auto-selected latest checkpoint: {ckpt_path}")
    elif not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)

    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = f'{output_dir}/{spk_name}/output.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        sentences = split_sentences(text)
        if len(sentences) == 1:
            audio = model.infer(sentences[0], spk_id=spk_id, language=language)
        else:
            audio_segments = []
            for sentence in sentences:
                segment = model.infer(sentence, spk_id=spk_id, language=language)
                audio_segments.append(segment)
            audio = np.concatenate(audio_segments, axis=-1)

        sf.write(save_path, audio, samplerate=model.hps.data.sampling_rate)
        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    main()

# import os
# import click
# from melo.api import TTS

    
    
# @click.command()
# # @click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
# @click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file or directory")

# @click.option('--text', '-t', type=str, default=None, help="Text to speak")
# @click.option('--language', '-l', type=str, default="EN", help="Language of the model")
# @click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
# def main(ckpt_path, text, language, output_dir):
#     if ckpt_path is None:
#         raise ValueError("The model_path must be specified")

#     # 🔧 Auto-select latest checkpoint if a directory is passed
#     if os.path.isdir(ckpt_path):
#         ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith("G_") and f.endswith(".pth")]
#         if not ckpt_files:
#             raise FileNotFoundError(f"No G_*.pth files found in {ckpt_path}")
#         ckpt_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
#         ckpt_path = os.path.join(ckpt_path, ckpt_files[-1])
#         print(f"✅ Auto-selected latest checkpoint: {ckpt_path}")
#     elif not os.path.isfile(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

#     config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
#     model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)

#     for spk_name, spk_id in model.hps.data.spk2id.items():
#         save_path = f'{output_dir}/{spk_name}/output.wav'
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         model.tts_to_file(text, spk_id, save_path)

# if __name__ == "__main__":
#     main()
