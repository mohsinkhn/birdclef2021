from copy import deepcopy

import numpy as np
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from torch import nn
import torch.nn.functional as F

from src.inference import get_models_preds, get_score, sigmoid


class SelfAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, gamma=1.0, dropout=0.2):
        super().__init__()
        self.gamma = gamma
        self.drop = nn.Dropout(dropout)
        self.fx = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.gx = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1)
        # self.hx = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.vx = nn.Conv1d(mid_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        fi = torch.softmax(torch.tanh(self.fx(self.drop(x))), -1)  # B x C2 x W2
        gi = torch.sigmoid(self.gx(self.drop(x)))  # B x C2 x W2
        aij = fi * gi  # B x C2 x W2
        oi = self.vx(aij)  # B x C1 x W2
        return x + self.gamma * oi


class SedAttnCNN(nn.Module):
    def __init__(
        self,
        backbone="densenet121",
        pretrained=True,
        dropout=0.5,
        fc_channels=1024,
        attn_channels=2048,
        attn_gamma=1.0,
        attn_dropout=0.1,
        num_classes=398
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.backbone.num_features, fc_channels)
        self.attn = SelfAttention(self.backbone.num_features, attn_channels, attn_gamma, attn_dropout)
        self.clf = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # B, C, W, H --> B, C, W1, H1
        x = torch.mean(x, dim=3)  # B, C, W1
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        # x = x.transpose(1, 2)
        # x = F.relu_(self.fc1(x))
        # x = x.transpose(1, 2)
        # x = self.drop(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.clf(x.transpose(1, 2))
        return torch.sum(x, 1)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""
    return output


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class SedAttnCNN2(nn.Module):
    def __init__(
        self,
        backbone="densenet121",
        pretrained=True,
        dropout=0.5,
        fc_channels=1024,
        attn_channels=2048,
        attn_gamma=1.0,
        attn_dropout=0.1,
        num_classes=398
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(1024, num_classes, activation='sigmoid')
        self.interpolate_ratio = 32  # Downsampled ratio

    def forward(self, x):
        x = self.backbone.forward_features(x)  # B, C, W, H --> B, C, W1, H1
        x = torch.mean(x, dim=3)  # B, C, W1
        b, c, frames_num = x.shape

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        # x = x.transpose(1, 2)
        # x = F.relu_(self.fc1(x))
        # x = x.transpose(1, 2)
        # x = self.drop(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        frame_shape = framewise_output.shape
        clip_shape = clipwise_output.shape
        output_dict = {
            'framewise_output': framewise_output.reshape(b, c, frame_shape[1], frame_shape[2]),
            'clipwise_output': clipwise_output.reshape(b, c, clip_shape[1]),
        }

        return output_dict


class SedCNN(nn.Module):
    def __init__(
        self,
        backbone="densenet121",
        pretrained=True,
        dropout=0.5,
        fc_channels=1024,
        attn_channels=2048,
        attn_gamma=1.0,
        attn_dropout=0.1,
        num_classes=398
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # B, C, W, H --> B, C, W1, H1
        x = torch.mean(x, dim=3)  # B, C, W1
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = torch.mean(x, 2)
        x = self.drop(x)
        return self.clf(x)


class BirdModel(pl.LightningModule):
    def __init__(self, config, stage):
        super().__init__()
        self.config = config
        self.stage = stage
        self.model = self.config.get_model(self.stage)
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        y = batch["labels"]
        x = batch["x"]
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        f1, prec, rec = self._get_metrics(y, torch.sigmoid(logits))
        self.log("train_f1", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        self.log("train_precision", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        self.log("train_recall", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["labels"]
        x = batch["x"]
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, on_epoch=True, logger=True)
        return {"loss": loss, "y_hat": torch.sigmoid(logits), "y": y}

    def validation_epoch_end(self, outputs):
        y_hats, ys = [], []
        for output in outputs:
            y_hats.append(output["y_hat"])
            ys.append(output["y"])
        y_hats = torch.cat(y_hats)
        ys = torch.cat(ys)
        opt_thresh, opt_f1, opt_prec, opt_rec = 0, 0, 0, 0
        for thresh in np.arange(0.1, 0.25, 0.025):
            f1, prec, rec = self._get_metrics(ys, y_hats, thresh)
            if f1 > opt_f1:
                opt_thresh = thresh
                opt_f1 = f1
                opt_prec = prec
                opt_rec = rec
        test_info = pd.read_csv('data/train_soundscape_labels.csv')
        filenames = list(Path("data/train_soundscapes").glob("*.ogg"))
        filename_map = {"_".join(f.stem.split("_")[:2]): str(f) for f in filenames}
        test_info["filepaths"] = test_info.row_id.apply(lambda x: filename_map["_".join(x.split("_")[:2])])
        config = deepcopy(self.config.config)
        config.width = config.sr // config.hop_length * 5
        _, probs,  _ = get_models_preds([self], [config], test_info.filepaths.unique(), test_info, 0.88, 0.95)
        test_score = 0
        test_thresh = 0
        for thresh in np.arange(0.0, 0.9, 0.025):
            score = get_score(sigmoid(np.mean(probs, 0)), test_info.birds.values, thresh, thresh)[0]
            if score > test_score:
                test_score = score
                test_thresh = thresh
        print(test_thresh, test_score)
        self.log("valid_f1", opt_f1)
        self.log("valid_precision", opt_prec)
        self.log("valid_recall", opt_rec)
        self.log("optimal_thresh", opt_thresh)
        self.log("test_f1", test_score)
        self.log("test_thresh", test_thresh)

    def configure_optimizers(self):
        optimizer_, optimizer_kws = self.config.get_optimizer(self.stage)
        params = list(self.model.named_parameters())

        def is_backbone(x):
            return "backbone" in x

        grouped_parameters = [
            {
                "params": [p for n, p in params if is_backbone(n)],
                "lr": optimizer_kws["lr"] * optimizer_kws.get("backbone_ratio", 0.1),
            },
            {"params": [p for n, p in params if not is_backbone(n)], "lr": optimizer_kws["lr"]},
        ]

        optimizer = optimizer_(grouped_parameters, weight_decay=optimizer_kws["weight_decay"])
        scheduler_, scheduler_kws, scheduler_type = self.config.get_scheduler(self.stage)
        scheduler = scheduler_(optimizer, **scheduler_kws)
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": self.config.config.training[self.stage].get("scheduler_type", "epoch")}],
        )

    def _get_metrics(self, y, y_hat, thresh=0.5, eps=1e-6):
        preds = (y_hat > thresh) * 1.0
        y = y >= self.config.config.secondary_confidence
        preds[preds.sum(1) == 0, -1] = 1.0
        prec = ((preds * y).sum() / (eps + preds.sum())).item()
        rec = ((preds * y).sum() / (eps + y.sum())).item()
        f1 = 2 * prec * rec / (eps + prec + rec)
        return f1, prec, rec


class BirdModel2(pl.LightningModule):
    def __init__(self, config, stage):
        super().__init__()
        self.config = config
        self.stage = stage
        self.model = self.config.get_model(self.stage)
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        y = batch["labels"]
        x = batch["x"]
        logits = self.model(x)['clipwise_output']
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        f1, prec, rec = self._get_metrics(y, torch.sigmoid(logits))
        self.log("train_f1", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        self.log("train_precision", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        self.log("train_recall", f1, on_epoch=True, logger=True, reduce_fx=torch.mean)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["labels"]
        x = batch["x"]
        logits = self.model(x)['clipwise_output']
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, on_epoch=True, logger=True)
        return {"loss": loss, "y_hat": torch.sigmoid(logits), "y": y}

    def validation_epoch_end(self, outputs):
        y_hats, ys = [], []
        for output in outputs:
            y_hats.append(output["y_hat"])
            ys.append(output["y"])
        y_hats = torch.cat(y_hats)
        ys = torch.cat(ys)
        opt_thresh, opt_f1, opt_prec, opt_rec = 0, 0, 0, 0
        for thresh in np.arange(0.1, 0.25, 0.025):
            f1, prec, rec = self._get_metrics(ys, y_hats, thresh)
            if f1 > opt_f1:
                opt_thresh = thresh
                opt_f1 = f1
                opt_prec = prec
                opt_rec = rec
        test_info = pd.read_csv('data/train_soundscape_labels.csv')
        filenames = list(Path("data/train_soundscapes").glob("*.ogg"))
        filename_map = {"_".join(f.stem.split("_")[:2]): str(f) for f in filenames}
        test_info["filepaths"] = test_info.row_id.apply(lambda x: filename_map["_".join(x.split("_")[:2])])
        config = deepcopy(self.config.config)
        config.width = config.sr // config.hop_length * 5
        _, probs,  _ = get_models_preds([self], [config], test_info.filepaths.unique(), test_info, 0.88, 0.95)
        test_score = 0
        test_thresh = 0
        for thresh in np.arange(0.0, 0.9, 0.025):
            score = get_score(sigmoid(np.mean(probs, 0)), test_info.birds.values, thresh, thresh)[0]
            if score > test_score:
                test_score = score
                test_thresh = thresh
        print(test_thresh, test_score)
        self.log("valid_f1", opt_f1)
        self.log("valid_precision", opt_prec)
        self.log("valid_recall", opt_rec)
        self.log("optimal_thresh", opt_thresh)
        self.log("test_f1", test_score)
        self.log("test_thresh", test_thresh)

    def configure_optimizers(self):
        optimizer_, optimizer_kws = self.config.get_optimizer(self.stage)
        params = list(self.model.named_parameters())

        def is_backbone(x):
            return "backbone" in x

        grouped_parameters = [
            {
                "params": [p for n, p in params if is_backbone(n)],
                "lr": optimizer_kws["lr"] * optimizer_kws.get("backbone_ratio", 0.1),
            },
            {"params": [p for n, p in params if not is_backbone(n)], "lr": optimizer_kws["lr"]},
        ]

        optimizer = optimizer_(grouped_parameters, weight_decay=optimizer_kws["weight_decay"])
        scheduler_, scheduler_kws, scheduler_type = self.config.get_scheduler(self.stage)
        scheduler = scheduler_(optimizer, **scheduler_kws)
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": self.config.config.training[self.stage].get("scheduler_type", "epoch")}],
        )

    def _get_metrics(self, y, y_hat, thresh=0.5, eps=1e-6):
        preds = (y_hat > thresh) * 1.0
        y = y >= self.config.config.secondary_confidence
        preds[preds.sum(1) == 0, -1] = 1.0
        prec = ((preds * y).sum() / (eps + preds.sum())).item()
        rec = ((preds * y).sum() / (eps + y.sum())).item()
        f1 = 2 * prec * rec / (eps + prec + rec)
        return f1, prec, rec

# def plot_labels(c, names=(), save_dir=Path(''), loggers=None):
#     # plot dataset labels
#     print('Plotting labels... ')
#     nc = int(c.max() + 1)  # number of classes

#     # matplotlib labels
#     matplotlib.use('svg')  # faster
#     ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
#     ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
#     ax[0].set_ylabel('instances')
#     if 0 < len(names) < 30:
#         ax[0].set_xticks(range(len(names)))
#         ax[0].set_xticklabels(names, rotation=90, fontsize=10)
#     else:
#         ax[0].set_xlabel('classes')
#     sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
#     sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

#     for a in [0, 1, 2, 3]:
#         for s in ['top', 'right', 'left', 'bottom']:
#             ax[a].spines[s].set_visible(False)

#     plt.savefig(save_dir / 'labels.jpg', dpi=200)
#     matplotlib.use('Agg')
#     plt.close()

#     # loggers
#     for k, v in loggers.items() or {}:
#         if k == 'wandb' and v:
#             v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)