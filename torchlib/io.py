import json
import yaml

import albumentations as A
from easydict import EasyDict
from timm import optim as toptim
import torch
from torch import optim
from torch.optim import lr_scheduler

from src import lightningmodels


def load_yaml(file):
    with open(file, 'r') as yfile:
        cfg = yaml.safe_load(yfile)
    return EasyDict(cfg)


def load_json(file):
    with open(file, 'r') as yfile:
        cfg = json.load(yfile)
    return EasyDict(cfg)


class ConfigParser:
    def __init__(self, config):
        self.config = load_yaml(config)

    def get_training_stages(self):
        return self.config.training.keys()

    def get_optimizer(self, stage):
        stage_params = self.config.training[stage]
        if hasattr(toptim, stage_params['optimizer']):
            optimizer_ = getattr(toptim, stage_params['optimizer'])
        else:
            optimizer_ = getattr(optim, stage_params['optimizer'])
        optimizer_kws_ = stage_params['optimizer_kwargs']
        return optimizer_, optimizer_kws_

    def get_scheduler(self, stage):
        stage_params = self.config.training[stage]
        scheduler_ = getattr(lr_scheduler, stage_params['scheduler'])
        scheduler_kws_ = stage_params['scheduler_kwargs']
        scheduler_type = stage_params['scheduler_type']
        return scheduler_, scheduler_kws_, scheduler_type

    def get_train_params(self, stage):
        stage_params = self.config.training[stage]
        num_epochs = stage_params['num_epochs']
        accumulation_steps = stage_params['accumulation_steps']
        grad_clip = stage_params['grad_clip']
        return num_epochs, accumulation_steps, grad_clip

    def get_loader_params(self, stage):
        stage_params = self.config.training[stage]
        batch_size = stage_params['batch_size']
        num_workers = stage_params['num_workers']
        return batch_size, num_workers

    def get_train_transforms(self, stage):
        return self.get_transforms(stage, 'train_tfms')

    def get_val_transforms(self, stage):
        return self.get_transforms(stage, 'val_tfms')

    def get_transforms(self, stage, flag):
        stage_params = self.config.training[stage]
        train_tfms = stage_params[flag]
        tfms_ = []
        for tfm, kws in train_tfms.items():
            tfm_ = getattr(A, tfm)(**kws)
            tfms_.append(tfm_)
        return A.Compose(tfms_)

    def get_model(self, stage):
        stage_params = self.config.training[stage]
        model_ = getattr(lightningmodels, stage_params.model.cls)
        model_kws = stage_params.model.kws
        model = model_(**model_kws)
        if stage_params.model.load_path != 'None':
            model.load_state_dict(torch.load(stage_params.model.load_path), strict=False)
        return model
