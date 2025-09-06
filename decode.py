import argparse
import os
import pprint
import random
import numpy as np
import soundfile as sf
import toml
import torch
import librosa
import torch.nn.modules.utils

from Libs.Casual_TFGridNet import Generator

from PAL.am_nld import AMNLD

from Libs.audio import get_comSTFT, get_comiSTFT
from Libs.utils import get_logger, load_obj
from scipy.io import loadmat

logger = get_logger(__name__)


def write_wav(fname, samps, fs=16000):
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)

    sf.write(fname, samps, fs)


def rms(audio, db=False):
    audio = np.asarray(audio)
    rms_value = np.sqrt(np.mean(audio ** 2))

    if db:
        return 20 * np.log10(rms_value + np.finfo(float).eps)
    else:
        return rms_value


def normalize(audio,
              target_level=-25,
              rms_ix_start=0,
              rms_ix_end=None,
              return_scalar=False):
    rms_value = rms(audio[rms_ix_start:rms_ix_end])
    scalar = 10 ** (target_level / 20) / (rms_value + np.finfo(float).eps)
    audio = audio * scalar

    if return_scalar:
        return audio, scalar
    else:
        return audio


class Decoder(object):
    def __init__(self,
                 generator,
                 gpuid=(7,),
                 cpt_dir=None,
                 config=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)

        self.gpuid = gpuid
        self.configs_signal = config['signal']
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.checkpoint = cpt_dir
        self.gen_params = sum([param.nelement() for param in generator.parameters()]) / 10.0 ** 6

        trd = loadmat("PAL/TRD_freq_16k_complex_norm.mat")["h_n"]
        self.trd = torch.as_tensor(trd, dtype=torch.complex64).to(self.device)
        h_all = loadmat("PAL/Hall_freq_32k.mat")["h_n"]
        self.h_all = torch.as_tensor(h_all, dtype=torch.complex64).to(self.device)

        self.modula = configs['net']['idt']['modula']
        pal = AMNLD(trd=self.trd,
                    h_all=self.h_all,
                    device=self.device,
                    sr=configs['signal']['idt']['sr'],
                    fs=configs['signal']['idt']['fs'],
                    win_size=configs['signal']['idt']['win_size'],
                    win_shift=configs['signal']['idt']['win_shift'],
                    fft_num=configs['signal']['idt']['fft_num'],
                    is_diff=configs['net']['idt']['is_diff'],
                    is_trd=configs['net']['idt']['is_trd'],
                    is_ssb=configs['net']['idt']['is_ssb'],
                    is_vf=configs['net']['idt']['is_vf'],
                    modula=configs['net']['idt']['modula']
                    )
        self.nnet = pal

        # set eval model
        cpt_fname = os.path.join(cpt_dir, 'checkpoint', "best_622.pt.tar")
        cpt = torch.load(cpt_fname, map_location="cpu")
        logger.info("Load checkpoint from {}, epoch {:d}".format(cpt_fname, cpt["epoch"]))

        generator.load_state_dict(cpt["generator_state_dict"])
        self.generator = generator.to(self.device)
        self.generator.eval()
        
        logger.info("Loading generator model to GPUs:{}, #param: {:.2f}M".format(gpuid, self.gen_params))
    
    def pal_real(self, x, x_len):
        """
        input:
            x: (B,T) 
        return:
            x_pal: (B,T)
        """
        x_mean = torch.mean(x, dim=-1)
        x = x - x_mean.unsqueeze(dim=-1)
        x_pal = self.nnet.tf_diffvf(x, x_len, self.device) 
        return x_pal

    def estimate(self, egs):
        with torch.no_grad():
            egs = load_obj(egs, self.device)

            batch_clean_stft = get_comSTFT(egs['mics'], self.configs_signal, self.device)
            batch_est_stft = self.generator(batch_clean_stft)
            
            est_wav = get_comiSTFT(batch_est_stft, max(egs['frame_mask_lists']), self.configs_signal, self.device)

            pal_rsc_wav = self.pal_real(est_wav, max(egs['frame_mask_lists']))
            
            return est_wav.cpu().numpy(), pal_rsc_wav.cpu().numpy()

def run(cpt_dir, config):  
    generator = Generator(n_fft=configs['signal']['fft_num'],
                        n_band=configs['signal']['band_split'],
                        n_layers=configs['net']['generator']['n_layers'],
                        lstm_hidden_units=configs['net']['generator']['lstm_hidden_units'],
                        attn_n_head=configs['net']['generator']['attn_n_head'],
                        attn_approx_qk_dim=configs['net']['generator']['attn_approx_qk_dim'],
                        emb_dim=configs['net']['generator']['emb_dim'],
                        emb_ks=configs['net']['generator']['emb_ks'],
                        emb_hs=configs['net']['generator']['emb_hs'],
                        activation=configs['net']['generator']['activation'],
                        eps=configs['net']['generator']['eps'])
    
    mic_inputs = [line.strip().split()[0] for line in open(config['path']['test']['scp_path'])]
    # random.shuffle(mic_inputs)

    decoder = Decoder(generator, cpt_dir=cpt_dir, config=config)

    for iteration, mic in enumerate(mic_inputs):
        mic_wav, sr0 = sf.read(mic)
        mic_wav = librosa.resample(mic_wav, orig_sr=sr0, target_sr=configs['signal']['sr'])

        mic_std_ = np.max(np.abs(mic_wav)) + 1e-7
        mic_wav = mic_wav / mic_std_

        egs = {'mics': torch.tensor(mic_wav, dtype=torch.float32).unsqueeze(0),
               'frame_mask_lists': torch.tensor([len(mic_wav)], dtype=torch.int)}

        est_wav, pal_rsc_wav = decoder.estimate(egs)
        est_wav = np.squeeze(est_wav, axis=0)
        pal_rsc_wav = np.squeeze(pal_rsc_wav, axis=0)

        assert len(est_wav) == len(mic_wav), "Est length must be equal to Mix length!"

        write_wav(os.path.join(cpt_dir, 'best', 'PAL_rsc',
                               os.path.basename(mic)
                               ),
                  pal_rsc_wav,
                  fs=configs['signal']['sr'])

        write_wav(os.path.join(cpt_dir, 'best', 'Pre',
                               os.path.basename(mic)
                               ),
                  est_wav,
                  fs=configs['signal']['sr'])
        write_wav(os.path.join(cpt_dir, 'best', 'Tar',
                               os.path.basename(mic)
                               ),
                  mic_wav * mic_std_,
                  fs=configs['signal']['sr'])
        logger.info("Compute on utterance: {:s}...".format(os.path.basename(mic)))

    logger.info("Compute over {0:d} utterances".format(iteration + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to start TFGridGAN_nld_inv.py decoding",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="Configs/train_config.toml",
                        help="Path to configs")
    parser.add_argument("--cpt_dir",
                        type=str,
                        required=False,
                        default='Exp/CasualTFGridNet_TFDiffVF_inv',
                        help="Name to datasets")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    configs = toml.load(args.config)
    logger.info(configs)

    run(args.cpt_dir, configs)
