import torch
import numpy as np
import librosa
from PAL.utils import stft, istft, freq_conv

import soundfile as sf
from scipy.io import loadmat

class AMNLD(object):
    def __init__(self,
                 trd,
                 h_all,
                 device,
                 sr: int = 32000,
                 fs: int = 16000,
                 win_size: int = 640,
                 win_shift: int = 320,
                 fft_num: int = 640,
                 is_diff: bool = True,
                 is_trd: bool = True,
                 is_ssb: bool = True,
                 is_vf: bool = True,
                 modula: float = 0.8,
                 ):
        self.trd = trd
        self.h_all = h_all
        self.device = device
        self.sr = sr
        self.fs = fs
        self.win_size = win_size
        self.win_shift = win_shift
        self.fft_num = fft_num
        self.is_diff = is_diff
        self.is_trd = is_trd
        self.is_ssb = is_ssb
        self.is_vf = is_vf
        self.modula = modula
        
    def tf_diffvf(self,
            x,
            x_len,
            device):
        if self.is_trd:
            x = freq_conv(self.trd, x, device)  # (B,T,F)  
        if not self.is_ssb:  
            x = self.modula * self.modula * torch.square(x) + 2 * self.modula * x
        # else:
        #     x_hil = hilbert(x)
        #     x = self.modula * self.modula * torch.square(x - 1j * x_hil) + 2 * self.modula * (x - 1j * x_hil)
        if self.is_diff and self.is_vf:
            X = stft(x, device, win_size=self.win_size, win_shift=self.win_shift, fft_num=self.win_size)
            X_gain = X * self.h_all
            est_pal_wav = istft(X_gain, device, sig_len=x_len, win_size=self.win_size, win_shift=self.win_shift, fft_num=self.win_size)
            est_pal_wav = est_pal_wav.real 
        else:
            est_pal_wav = x
        return est_pal_wav
    
