import toml
import torch
import random
import numpy as np
import torch.nn as nn
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset


configs = toml.load('Configs/train_config.toml')


class TrDataset(Dataset):
    def __init__(self):
        self.batch_size = configs['dataloader']['batch_size']

        start = 0
        minibatch = []
        file_list = [line.strip().split() for line in open(configs['path']['train']['scp_path'])]
        while True:
            end = min(len(file_list), start + self.batch_size)
            minibatch.append(file_list[start:end])
            start = end
            if end == len(file_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class CvDataset(Dataset):
    def __init__(self):
        self.batch_size = configs['dataloader']['batch_size']

        start = 0
        minibatch = []
        file_list = [line.strip().split() for line in open(configs['path']['val']['scp_path'])]
        while True:
            end = min(len(file_list), start + self.batch_size)
            minibatch.append(file_list[start:end])
            start = end
            if end == len(file_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class MyDataLoader(object):
    def __init__(self, dataset, **kw):
        self.data_loader = DataLoader(dataset=dataset,
                                      shuffle=True,
                                      pin_memory=configs['dataloader']['pin_memory'],
                                      num_workers=configs['dataloader']['num_workers'],
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        mics, targets, frame_mask_lists = generate_feats_labels(batch)

        return {'mics': mics,
                'targets': targets,
                'frame_mask_lists': frame_mask_lists}

    def get_data_loader(self):
        return self.data_loader


def generate_feats_labels(batch):
    batch = batch[0]
    chunk_length = int(configs['signal']['sr'] * configs['signal']['chunk_length'])

    mic_list, target_list, frame_mask_list = [], [], []
    for id in range(len(batch)):
        mic_file_name = '%s' % batch[id][0]
        target_file_name = '%s' % batch[id][1]

        mic_wav, sr0 = sf.read(mic_file_name)
        target_wav, _ = sf.read(target_file_name)

        mic_wav = librosa.resample(mic_wav, orig_sr=sr0, target_sr=configs['signal']['sr'])
        target_wav = librosa.resample(target_wav, orig_sr=sr0, target_sr=configs['signal']['sr'])

        if len(mic_wav) > chunk_length:
            wav_start = random.randint(0, len(mic_wav) - chunk_length)

            mic_wav = mic_wav[wav_start:wav_start + chunk_length]
            target_wav = target_wav[wav_start:wav_start + chunk_length]

        mic_std_ = np.std(mic_wav) + 1e-7
        # mic_std_ = np.max(np.abs(mic_wav)) + 1e-7
        mic_wav, target_wav = mic_wav / mic_std_, target_wav / mic_std_

        mic_list.append(torch.FloatTensor(mic_wav))
        target_list.append(torch.FloatTensor(target_wav))
        frame_mask_list.append(len(mic_wav))

    mic_list = nn.utils.rnn.pad_sequence(mic_list, batch_first=True)
    target_list = nn.utils.rnn.pad_sequence(target_list, batch_first=True)

    mic_list, target_list = mic_list.contiguous(), target_list.contiguous()

    return mic_list, target_list, frame_mask_list


if __name__ == '__main__':
    tr_dataset = TrDataset()
    tr_loader = MyDataLoader(dataset=tr_dataset,
                             batch_size=1)
    data = next(iter(tr_loader.get_data_loader()))
