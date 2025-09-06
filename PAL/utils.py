import torch
import numpy as np

EPSILON = 1e-8
def stft(batch_sig_wav, device, onesided=False, win_size=640, win_shift=320, fft_num=640):
    batch_sig_stft = torch.stft(batch_sig_wav,
                                n_fft=fft_num,
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                onesided=onesided,
                                window=torch.hann_window(win_size).to(device)
                                )
    batch_sig_stft = batch_sig_stft.permute(0, 2, 1).contiguous()  # （B,T,F）
    return batch_sig_stft


def istft(batch_sig_stft, device, sig_len, onesided=False, win_size=640, win_shift=320, fft_num=640):
    batch_sig_stft = batch_sig_stft.permute(0, 2, 1)
    sig_utt = torch.istft(batch_sig_stft,
                          n_fft=fft_num,
                          hop_length=win_shift,
                          win_length=win_size,
                          window=torch.hann_window(win_size).to(device),
                          length=sig_len,
                          return_complex=True,
                          onesided=onesided)  # (B,L)
    return sig_utt

def freq_conv(H, x, device):
    win_size = H.shape[-1]
    hop_size = win_size // 2

    x = x.real
    X = stft(x, device, win_size=win_size, win_shift=hop_size, fft_num=win_size)  # (B,T,F)
    Y = X * H
    y = istft(Y, device, sig_len=x.shape[-1], win_size=win_size, win_shift=hop_size, fft_num=win_size)
    y = y.real

    return y

def freq_diff(x, device, fs, n=2, win_size=640, win_shift=320, fft_num=640):
    """
        Args:
            fs : sampling rate of x.
            n (int): order of derivative.
    """
    x = torch.as_tensor(x)
    X = stft(x, device, win_size=win_size, win_shift=win_shift, fft_num=fft_num)
    f = torch.as_tensor(np.fft.fftfreq(win_size, d=1 / fs)).to(device)
    X_gain = (2j * np.pi * f) ** n * X
    x_gain = istft(torch.as_tensor(X_gain), device, sig_len=x.shape[-1], win_size=win_size, win_shift=win_shift, fft_num=fft_num)

    return x_gain