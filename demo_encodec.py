from audiolm_pytorch import EncodecWrapper

import math
import wave
import struct
import os
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio
from torchaudio.functional import resample

encodec = EncodecWrapper()

# audio = torchaudio.load("./dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac")[0]
audio = torchaudio.load("极MMM_JPVer_demo0730_01.wav")[0]

audio = resample(audio, 44100, 24000) 
emb, codes, _ = encodec(audio, return_encoded=True)
recon = encodec.decode_from_codebook_indices(codes)
audio_recon = recon.detach().cpu().squeeze(1)
torchaudio.save("ji_encodec_recon.flac", audio_recon, 24000)
