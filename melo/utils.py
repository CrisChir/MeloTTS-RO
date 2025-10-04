import os, glob, argparse, logging, json, subprocess, numpy as np
from scipy.io.wavfile import read
import torch, torchaudio, librosa
from melo import commons

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict: v = HParams(**v)
            self[k] = v
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def __repr__(self): return self.__dict__.__repr__()

logger = logging.getLogger(__name__)

def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="JSON file for configuration")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    args, unknown = parser.parse_known_args()
    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)
    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r", encoding='utf-8') as f: data = f.read()
        with open(config_save_path, "w", encoding='utf-8') as f: f.write(data)
    else:
        with open(config_save_path, "r", encoding='utf-8') as f: data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def load_wav_to_torch_librosa(full_path, sr):
    audio, _ = librosa.load(full_path, sr=sr, mono=True)
    return torch.FloatTensor(audio.astype(np.float32)), sr

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(f"{source_dir} is not a git repository...")
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(f"git hash values are different. {saved_hash[:8]} != {cur_hash[:8]}")
    else:
        open(path, "w").write(cur_hash)

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG); h.setFormatter(formatter); logger.addHandler(h)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG); stream_handler.setFormatter(formatter); logger.addHandler(stream_handler)
    return logger

# import os
# import glob
# import argparse
# import logging
# import json
# import subprocess
# import numpy as np
# from scipy.io.wavfile import read
# import torch
# import torchaudio
# import librosa
# from melo import commons

# # The HParams class is needed by get_hparams
# class HParams:
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             if type(v) == dict:
#                 v = HParams(**v)
#             self[k] = v
#     def keys(self): return self.__dict__.keys()
#     def items(self): return self.__dict__.items()
#     def values(self): return self.__dict__.values()
#     def __len__(self): return len(self.__dict__)
#     def __getitem__(self, key): return getattr(self, key)
#     def __setitem__(self, key, value): return setattr(self, key, value)
#     def __contains__(self, key): return key in self.__dict__
#     def __repr__(self): return self.__dict__.__repr__()

# MATPLOTLIB_FLAG = False
# logger = logging.getLogger(__name__)

# # This is the function that train.py needs to start
# def get_hparams(init=True):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", type=str, required=True, help="JSON file for configuration")
#     parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
#     args, unknown = parser.parse_known_args()
#     model_dir = os.path.join("./logs", args.model)
#     os.makedirs(model_dir, exist_ok=True)
#     config_path = args.config
#     config_save_path = os.path.join(model_dir, "config.json")
#     if init:
#         with open(config_path, "r", encoding='utf-8') as f: data = f.read()
#         with open(config_save_path, "w", encoding='utf-8') as f: f.write(data)
#     else:
#         with open(config_save_path, "r", encoding='utf-8') as f: data = f.read()
#     config = json.loads(data)
#     hparams = HParams(**config)
#     hparams.model_dir = model_dir
#     return hparams

# # This is the other function that train.py needs
# def check_git_hash(model_dir):
#     source_dir = os.path.dirname(os.path.realpath(__file__))
#     if not os.path.exists(os.path.join(source_dir, ".git")):
#         logger.warn(f"{source_dir} is not a git repository, therefore hash value comparison will be ignored.")
#         return
#     cur_hash = subprocess.getoutput("git rev-parse HEAD")
#     path = os.path.join(model_dir, "githash")
#     if os.path.exists(path):
#         saved_hash = open(path).read()
#         if saved_hash != cur_hash:
#             logger.warn(f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)")
#     else:
#         open(path, "w").write(cur_hash)

# # This is the function your data_utils.py needs
# def load_wav_to_torch_librosa(full_path, sr):
#     audio_norm, _ = librosa.load(full_path, sr=sr, mono=True)
#     return torch.FloatTensor(audio_norm.astype(np.float32)), sr

# def load_filepaths_and_text(filename, split="|"):
#     with open(filename, encoding="utf-8") as f:
#         filepaths_and_text = [line.strip().split(split) for line in f]
#     return filepaths_and_text

# def get_logger(model_dir, filename="train.log"):
#     global logger
#     logger = logging.getLogger(os.path.basename(model_dir))
#     logger.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
#     if not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
#     h = logging.FileHandler(os.path.join(model_dir, filename))
#     h.setLevel(logging.DEBUG); h.setFormatter(formatter)
#     logger.addHandler(h)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.DEBUG); stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#     return logger

# # Other helper functions from the original file
# def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
#     assert os.path.isfile(checkpoint_path)
#     checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
#     iteration = checkpoint_dict.get("iteration", 0)
#     learning_rate = checkpoint_dict.get("learning_rate", 0.)
#     if (optimizer is not None and not skip_optimizer and "optimizer" in checkpoint_dict):
#         optimizer.load_state_dict(checkpoint_dict["optimizer"])
#     saved_state_dict = checkpoint_dict["model"]
#     if hasattr(model, "module"):
#         state_dict = model.module.state_dict()
#     else:
#         state_dict = model.state_dict()
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         try:
#             new_state_dict[k] = saved_state_dict[k]
#         except:
#             logger.info(f"{k} is not in the checkpoint")
#             new_state_dict[k] = v
#     if hasattr(model, "module"):
#         model.module.load_state_dict(new_state_dict, strict=False)
#     else:
#         model.load_state_dict(new_state_dict, strict=False)
#     logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
#     return model, optimizer, learning_rate, iteration
