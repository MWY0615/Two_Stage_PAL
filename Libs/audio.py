import torch

EPSILON = 1e-8

def get_comSTFT(batch_mic_wav, signal_configs, device):
    win_size = int(signal_configs['sr'] * signal_configs['win_size'])
    win_shift = int(signal_configs['sr'] * signal_configs['win_shift'])
    band_num = signal_configs['fft_num'] // 2 // signal_configs['band_split'] + 1

    batch_mic_stft = torch.stft(batch_mic_wav,
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                window=torch.hann_window(win_size).to(device))
    batch_mic_stft = torch.view_as_real(batch_mic_stft)  # (B,F,T,2)
    batch_mic_mag, batch_mic_phase = (torch.norm(batch_mic_stft, dim=-1) + EPSILON)** signal_configs['beta'], \
        torch.atan2(batch_mic_stft[..., -1]  + EPSILON, batch_mic_stft[..., 0] + EPSILON)
    batch_mic_stft = torch.stack([batch_mic_mag * torch.cos(batch_mic_phase),
                                  batch_mic_mag * torch.sin(batch_mic_phase)], dim=1).transpose(2, 3)  # (B, 2，T，F)

    return batch_mic_stft


def get_comiSTFT(batch_mic_stft, length, signal_configs, device):
    win_size = int(signal_configs['sr'] * signal_configs['win_size'])
    win_shift = int(signal_configs['sr'] * signal_configs['win_shift'])
    band_num = signal_configs['fft_num'] // 2 // signal_configs['band_split'] + 1
    overlap_num = band_num * signal_configs['band_split'] - signal_configs['fft_num'] // 2 - 1

    batch_mic_mag, batch_mic_phase = torch.norm(batch_mic_stft, dim=1) ** (1 / signal_configs['beta']), \
        torch.atan2(batch_mic_stft[:, -1, ...] + EPSILON, batch_mic_stft[:, 0, ...] + EPSILON)  # (B, T, F) 
    batch_mic_stft = torch.stack((batch_mic_mag * torch.cos(batch_mic_phase),
                                  batch_mic_mag * torch.sin(batch_mic_phase)), dim=-1).transpose(1, 2)  # (B, F, T, 2) 
    batch_mic_stft = torch.view_as_complex(batch_mic_stft)  #(B, F, T)

    batch_mic_wav = torch.istft(batch_mic_stft,
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=False,
                                window=torch.hann_window(win_size).to(device),
                                length=length)  # (B,L)

    return batch_mic_wav
