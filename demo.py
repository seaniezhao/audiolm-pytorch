# imports
import math
import wave
import struct
import os
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from audiolm_pytorch import EncodecWrapper

from torch import nn
import torch
import torchaudio

dataset_folder = './dev-clean'
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer


encodec = EncodecWrapper()


# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
if not os.path.isdir("hubert"):
  os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
  urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
  urllib.request.urlretrieve(hubert_quantizer_download, f"./{hubert_quantizer}")

wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()


trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 32,
    data_max_length = 320 * 32,
    num_train_steps = 100000
)

trainer.train()




coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
).cuda()

# trainer = CoarseTransformerTrainer(
#     transformer = coarse_transformer,
#     codec = encodec,
#     wav2vec = wav2vec,
#     folder = dataset_folder,
#     batch_size = 32,
#     data_max_length = 320 * 32,
#     save_results_every = 500,
#     save_model_every = 1000,
#     num_train_steps = 10000
# )
# # NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# # adjusting save_*_every variables for the same reason

# trainer.train()



fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
).cuda()

# trainer = FineTransformerTrainer(
#     transformer = fine_transformer,
#     codec = encodec,
#     folder = dataset_folder,
#     batch_size = 32,
#     data_max_length = 320 * 32,
#     num_train_steps = 10000
# )
# # NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# # adjusting save_*_every variables for the same reason

# trainer.train()



# # -------------------------------------------------------------------

# semantic_transformer.load('./pt_sem/semantic.transformer.9000.pt')
# coarse_transformer.load('./pt_coarse/coarse.transformer.6000.pt')
# fine_transformer.load('./pt_fine/fine.transformer.9000.pt')


# # Everything together
# audiolm = AudioLM(
#     wav2vec = wav2vec,
#     codec = encodec,
#     semantic_transformer = semantic_transformer,
#     coarse_transformer = coarse_transformer,
#     fine_transformer = fine_transformer
# ).cuda()

# generated_wav = audiolm(batch_size = 1)
# output_path = "out.wav"
# sample_rate = 24000
# torchaudio.save(output_path, generated_wav.cpu(), 24000)