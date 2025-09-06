import os
import time

import hdf5storage
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Libs.Casual_TFGridNet import Generator
from Libs.audio import get_comSTFT, get_comiSTFT
from Libs.loss import ComMagEuclideanLoss, SISDRLoss
from Libs.utils import get_logger, load_obj
from PAL.am_nld import AMNLD

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scipy.io import loadmat


EPSILON = 1e-10


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.generator_loss = []
        self.discriminator_loss = []
        self.timer = SimpleTimer()

    def add(self, generator_loss):
        self.generator_loss.append(generator_loss)

        N = len(self.generator_loss)
        if not N % self.period:
            generator_avg = sum(self.generator_loss[-self.period:]) / self.period
            if self.logger:
                self.logger.info("Processed {:d} batches (generator loss = {:+.4f}..."
                             .format(N, generator_avg))

    def report(self, details=False):
        N = len(self.generator_loss)

        if details:
            generator_str = ",".join(map(lambda f: "{:.2f}".format(f), self.generator_loss))
            if self.logger:
                self.logger.info("Generator Loss on {:d} batches: {}".format(N, generator_str))
        return {"generator_loss": sum(self.generator_loss) / N,
                "batches": N,
                "cost": self.timer.elapsed()}


class Trainer(object):
    def __init__(self,
                 rank,
                 gpuid,
                 configs=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)

        self.gpuid = gpuid
        self.rank = rank

        self.device = torch.device("cuda:{}".format(gpuid[rank]))

        self.configs_signal = configs['signal']
        self.checkpoint = configs['save']['save_filename']
        self.clip_norm = configs['optimizer']['gradient_norm']
        self.logging_period = configs['optimizer']['logging_period']
        self.no_impr = configs['optimizer']['early_stop_freq']
        self.resume = configs['path']['resume_filename']
        self.optimizer = configs['optimizer']['name']
        self.win_shift = int(configs['signal']['sr'] * configs['signal']['win_shift'])
        self.win_size = int(configs['signal']['sr'] * configs['signal']['win_size'])
        self.fft_num = int(configs['signal']['fft_num'])
        self.cur_epoch = 0

        trd = loadmat("PAL/TRD_freq_16k_complex_norm.mat")["h_n"]
        self.trd = torch.as_tensor(trd, dtype=torch.complex64).to(self.device)
        h_all = loadmat("PAL/Hall_freq_16k_complex_norm.mat")["h_n"]
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

        self.generator_params = sum([param.nelement() for param in generator.parameters()]) / 10.0 ** 6

        self.com_mag_loss = ComMagEuclideanLoss(alpha=0.5, l_type="L2")
        self.sisdr_loss = SISDRLoss(eps=configs["net"]["generator"]["eps"],
                                    zero_mean=configs["loss_function"]["zero_mean"],
                                    scale_label=configs["loss_function"]["scale_label"])

        os.makedirs(os.path.join(self.checkpoint, 'checkpoint'), exist_ok=True)

        self.logger = get_logger(os.path.join(self.checkpoint, "trainer_resetlr.log"), file=True) if self.rank == 0 else None

        if len(gpuid) > 1:
            generator = DistributedDataParallel(generator.to(self.device), find_unused_parameters=True)
           
        if self.resume:
            if not os.path.exists(self.resume):
                raise FileNotFoundError("Could not find resume checkpoint: {}".format(self.resume))
            cpt = torch.load(self.resume, map_location=self.device)
            self.cur_epoch = cpt["epoch"]

            if self.logger:
                self.logger.info("Resume from checkpoint {}: epoch {:d}".format(self.resume, self.cur_epoch))

            # load generator
            generator.load_state_dict(cpt["generator_state_dict"])
            self.generator = generator.to(self.device)
            self.generator_optimizer = self.create_optimizer(self.generator.parameters(),
                                                             self.optimizer,
                                                             configs['optimizer']['generator'],
                                                            #  state=cpt["generator_optim_state_dict"]
                                                            )
        else:
            self.generator = generator.to(self.device)
            self.generator_optimizer = self.create_optimizer(self.generator.parameters(),
                                                             self.optimizer,
                                                             configs['optimizer']['generator'])

        self.generator_scheduler = ReduceLROnPlateau(self.generator_optimizer,
                                                     mode="min",
                                                     factor=configs['optimizer']['generator']['factor'],
                                                     patience=configs['optimizer']['generator']['halve_freq'],
                                                     min_lr=configs['optimizer']['generator']['min_lr'],
                                                     verbose=True)

        # logging
        if self.logger:
            self.logger.info("Generator summary:\n{}".format(generator))
            self.logger.info("Loading model to GPUs:{}, #generator param: {:.2f}M"
                            .format(gpuid, self.generator_params))
            if self.clip_norm:
                self.logger.info("Gradient clipping by {}, default L2".format(self.clip_norm))

    def save_checkpoint(self, name, best=True):
        cpt = {"epoch": self.cur_epoch,
               "generator_state_dict": self.generator.state_dict(),
               "generator_optim_state_dict": self.generator_optimizer.state_dict(),
               }

        torch.save(cpt, os.path.join(self.checkpoint, 'checkpoint', "{0}.pt.tar"
                                     .format("best" if best else 'epoch_' + name)))

    def create_optimizer(self, parameters, optimizer, config, state=None):
        supported_optimizer = {"adam": torch.optim.Adam,
                               "adamw": torch.optim.AdamW, }

        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](parameters,
                                             lr=config["lr"],
                                             betas=(config["beta1"], config["beta2"]),
                                             weight_decay=config["weight_decay"])
        if self.logger:
            self.logger.info("Create optimizer {0}: {1}".format(optimizer, config))

        if state is not None:
            opt.load_state_dict(state)
            if self.logger:
                self.logger.info("Load optimizer state dict from checkpoint")

        return opt

    def generator_compute_loss(self, egs):
        raise NotImplementedError

    def discriminator_compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        
        self.generator.train()
        if self.logger:
            self.logger.info("Set train mode...")
        reporter = ProgressReporter(self.logger, period=self.logging_period)


        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.generator_optimizer.zero_grad()
            generator_loss = self.generator_compute_loss(egs)
            generator_loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.generator.parameters(), self.clip_norm)
            self.generator_optimizer.step()
            
            reporter.add(generator_loss.item())

        return reporter.report()

    def eval(self, data_loader):
        if self.logger:
            self.logger.info("Set eval mode...")
        self.generator.eval()
        
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with torch.no_grad():
            for egs in data_loader:
                # load to gpu
                egs = load_obj(egs, self.device)

                generator_loss = self.generator_compute_loss(egs)
                reporter.add(generator_loss.item())

        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=100):
        with (torch.cuda.device(self.device)):
            stats = dict()
            generator_no_impr = 0

            if self.rank == 0:
                self.save_checkpoint(name=str(self.cur_epoch), best=False)
            cv = self.eval(dev_loader)

            generator_best_loss = cv["generator_loss"]
            if self.logger:
                self.logger.info("START FROM EPOCH {:d}, GENERATOR LOSS = {:.4f}"
                             .format(self.cur_epoch, generator_best_loss))

            self.generator_scheduler.best = generator_best_loss

            train_epoch, val_epoch = [], [[cv["generator_loss"]]]
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1

                cur_generator_lr = self.generator_optimizer.param_groups[0]["lr"]
                stats["title"] = ("Loss(time/N, generator lr={:.3e}) - Epoch {:2d}:"
                                  .format(cur_generator_lr, self.cur_epoch))

                tr = self.train(train_loader)
                stats["tr"] = ("train = {:+.4f}/({:.2f}m/{:d})".
                               format(tr["generator_loss"], tr["cost"], tr["batches"]))
                train_epoch.append([tr["generator_loss"]])

                cv = self.eval(dev_loader)
                stats["cv"] = ("dev = {:+.4f}/({:.2f}m/{:d})"
                               .format(cv["generator_loss"], cv["cost"], cv["batches"]))
                val_epoch.append([cv["generator_loss"]])

                stats["scheduler"] = ""
                if cv["generator_loss"] > generator_best_loss:
                    generator_no_impr += 1
                    stats["scheduler"] = ("| no impr, best generator= {:.4f}"
                                          .format(self.generator_scheduler.best))
                else:
                    generator_best_loss = cv["generator_loss"]
                    generator_no_impr = 0
                    if self.rank == 0:
                        self.save_checkpoint(name=str(self.cur_epoch), best=True)
                if self.logger:
                    self.logger.info("{title} {tr} | {cv} {scheduler}".format(**stats))

                self.generator_scheduler.step(cv["generator_loss"])

                if self.rank == 0:
                    self.save_checkpoint(name=str(self.cur_epoch), best=False)
                if generator_no_impr == self.no_impr and self.logger:
                    self.logger.info("Stop training cause no impr for {:d} epochs".format(generator_no_impr))
                    break
            if self.logger:
                self.logger.info("Training for {:d}/{:d} epoches done!".format(self.cur_epoch, num_epochs))
                hdf5storage.savemat(os.path.join(self.checkpoint, 'loss_mat'), {'train': train_epoch, 'val': val_epoch})

        dist.destroy_process_group()


class SISNRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SISNRTrainer, self).__init__(*args, **kwargs)

    def generator_compute_loss(self, egs):
        generator_loss = 0.

        frame_list = [_ // (self.win_size // 2) + 1 for _ in egs['frame_mask_lists']]

        batch_est_stft = get_comSTFT(egs['mics'], self.configs_signal, self.device)
        batch_est_stft = self.generator(batch_est_stft)

        batch_est_wav = get_comiSTFT(batch_est_stft, max(egs['frame_mask_lists']), self.configs_signal, self.device) 
        pal_real = self.pal_real(batch_est_wav, max(egs['frame_mask_lists']))
        pal_real_stft = get_comSTFT(pal_real, self.configs_signal, self.device)

        batch_tar_stft = get_comSTFT(egs["targets"], self.configs_signal, self.device)

        cm_loss = self.com_mag_loss(pal_real_stft, batch_tar_stft, frame_list)
        sisdr_loss = self.sisdr_loss(pal_real, egs['targets'], egs['frame_mask_lists'])

        generator_loss += cm_loss + 0.2 * sisdr_loss

        total_loss = generator_loss

        return total_loss

    
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
    
    


    