import argparse
import gc
import json
import os
import random

# from logzero import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.dataset import MelspecDataset, TorchAudioValDataSet
from src.lightningmodels import BirdModel
from torchlib.io import ConfigParser, load_yaml
from src.imbalanced_dataset_sampler import ImbalancedDatasetSampler


def seed_everything(seed):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # set True to be faster
    print(f"Setting all seeds to be {seed} to reproduce...")


seed_everything(786)


def train_one_fold(df_tr, df_vl, config, stage, logdir, device, wandb_logger, resume_checkpoint):
    # Transforms
    # logging.info("Initializing transforms")

    # Dataset and Dataloaders
    # logging.info("Initializing dataloaders")
    tr_ds = MelspecDataset(
        df_tr.filepaths.values,
        df_tr.labels.values,
        df_tr.secondary_labels.values,
        config.config,
        None,
        True,
        True,
        True,
    )
    batch_size, num_workers = config.get_loader_params(stage)
    if config.config.get('sampler', '') == 'balanced':
        sampler = ImbalancedDatasetSampler(tr_ds)
        tr_dl = DataLoader(
            tr_ds,
            # shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler
        )
    else:
        tr_dl = DataLoader(
            tr_ds,
            shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    vl_ds = MelspecDataset(
        df_vl.filepaths.values,
        df_vl.labels.values,
        df_vl.secondary_labels.values,
        config.config,
        None,
        True,
        True,
        True,
    )
    batch_size, num_workers = config.get_loader_params(stage)
    vl_dl = DataLoader(
        vl_ds,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # sampler=sampler
    )

    # Model
    model = BirdModel(config, stage)
    checkpoint = ModelCheckpoint(
        dirpath=logdir, filename="{epoch}-{valid_f1:.3f}", monitor="valid_f1", mode="max", verbose=True
    )
    # average = StochasticWeightAveraging(30)
    # logging.info("Starting training...")
    num_epochs, accumulation_steps, clip_grad_norm = config.get_train_params(stage)
    trainer = pl.Trainer(
        gpus=[DEVICE],
        max_epochs=num_epochs,
        logger=wandb_logger,
        accumulate_grad_batches=accumulation_steps,
        gradient_clip_val=clip_grad_norm,
        weights_save_path=logdir,
        resume_from_checkpoint=resume_checkpoint,
        # limit_train_batches=0.1,
        # accelerator="ddp",
        callbacks=[checkpoint]  # , average]
        # stochastic_weight_avg=True,
        # precision=16,
        # amp_level='O1'
    )
    trainer.fit(model, tr_dl, vl_dl)
    return model, tr_dl, vl_dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--splitid", default='', type=str)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--resumefromcheckpoint")
    args = parser.parse_args()
    DEVICE = args.device
    FOLD = args.fold
    config = ConfigParser(args.config)
    df_tr = pd.read_parquet(f"data/tr{args.splitid}_{FOLD}.pq")
    df_vl = pd.read_parquet(f"data/vl{args.splitid}_{FOLD}.pq")

    df_tr["secondary_labels"] = df_tr.apply(lambda x: [code for code in x['all_labels'] if code not in x['labels']], axis=1)
    df_vl["secondary_labels"] = df_vl.apply(lambda x: [code for code in x['all_labels'] if code not in x['labels']], axis=1)

    if FOLD == -1:
        df_tr = pd.concat([df_tr, df_vl])

    device = f"cuda:{args.device}"
    exp_dir = Path("logs")
    logdir = exp_dir / config.config.run_name / f"fold_{FOLD}"
    logdir.mkdir(exist_ok=True, parents=True)
    with open(str(logdir / "config.json"), "w") as fp:
        json.dump(config.config, fp)
    wandb_logger = WandbLogger(project="birdclef2021", name=config.config.run_name + f"fold_{FOLD}")
    wandb_logger.log_hyperparams(config.config)
    stages = config.get_training_stages()
    for stage in stages:
        model, _, _ = train_one_fold(
            df_tr, df_vl, config, stage, logdir, device, wandb_logger, args.resumefromcheckpoint
        )
        del model
        gc.collect()
    # wandb_logger.finish()
