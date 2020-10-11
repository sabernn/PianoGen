# import import_ipynb

from midi2seq import *
import torch
import torch.nn as nn
from torch.utils.data import *
# from torchnlp.download import download_file_maybe_extract
# import torchnlp.download

from hw1 import Composer

# INPUT DATA 
bsz=32
epoch=1

piano_seq = torch.from_numpy(process_midi_seq(all_midis=None, datadir='./data/good', n=20, maxlen=50))

loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=4, drop_last=True)

cps=Composer()

for i in range(epoch):
    for x in loader:
        x[0] = torch.cat(x).view(len(x[0]), 1, -1)
        cps.train(x[0].long())

