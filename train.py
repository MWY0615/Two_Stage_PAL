import argparse
import pprint

import toml

from Libs.Casual_TFGridNet import Generator
from Libs.dataset import TrDataset, CvDataset, MyDataLoader
from Libs.trainer import SISNRTrainer
from Libs.utils import dump_json, get_logger

from torch.distributed import init_process_group
import torch.multiprocessing as mp

logger = get_logger(__name__)

import os

def run(rank, config):
    gpuids = tuple(config['gpu']['gpu_ids'])
    for conf, fname in zip([config['net'], config], ["net_config.json", "config.json"]):
        dump_json(conf, config['save']['save_filename'], fname)
    
    if len(gpuids) > 1:
        init_process_group(backend="nccl", init_method="tcp://localhost:54321", world_size=1 * len(gpuids), rank=rank)

    tr_dataset = TrDataset()
    cv_dataset = CvDataset()
    tr_loader = MyDataLoader(dataset=tr_dataset,
                             batch_size=1).get_data_loader()
    cv_loader = MyDataLoader(dataset=cv_dataset,
                             batch_size=1).get_data_loader()

    trainer = SISNRTrainer(rank=rank,
                         gpuid=gpuids,
                         configs=config)
    trainer.run(tr_loader, cv_loader, num_epochs=config['optimizer']['epochs'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to start GAN training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="Configs/train_config.toml",
                        help="Path to Configs")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    configs = toml.load(args.config)
    
    print(configs)
    print(len(configs))

    gpuids = tuple(configs['gpu']['gpu_ids'])
    print(gpuids)
    if len(gpuids) > 1:
        mp.spawn(run, args=(configs,), nprocs=len(gpuids))
    else:
        run(0, configs)
