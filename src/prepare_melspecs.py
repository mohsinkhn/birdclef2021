import librosa
from easydict import EasyDict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

tqdm.pandas()

from src.constants import CODE2INT, TRAIN_CLIPS


def get_melspec(filepath, config):
    y, _ = librosa.load(filepath, sr=config.sr, mono=True, res_type="kaiser_fast")
    melspec = librosa.feature.melspectrogram(
        y,
        sr=config.sr,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
    )
    return melspec.astype(np.float32)


if __name__ == "__main__":
    # CONFIG = "configs/melconfig.yaml"
    # with open(CONFIG, "r") as fp:
    #     config = EasyDict(yaml.safe_load(fp))
    # DATA_ROOT = "data"
    # SAVE_PATH = "data/melspecs"

    # train = pd.read_csv(Path(DATA_ROOT) / "train_metadata.csv")
    # train["filepaths"] = train.apply(lambda x: f"data/train_short_audio/{x['primary_label']}/{x['filename']}", axis=1,)
    # filepaths = train.filepaths.values.tolist()
    # folders = train.primary_label.values.tolist()
    # for filepath, folder in tqdm(zip(filepaths, folders)):
    #     fp = Path(filepath)
    #     save_path = Path(SAVE_PATH) / folder
    #     save_path.mkdir(exist_ok=True, parents=True)
    #     save_file = str(save_path / f"{fp.stem}.npy")
    #     melspec = get_melspec(filepath, config)
    #     np.save(str(save_file), melspec)

    # SAVE_PATH = "data/noisespecs"

    # train = pd.read_csv("/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/freefield/labels.txt")
    # train = train.loc[train.hasbird == 0]
    # train["filepaths"] = train.itemid.apply(lambda x: f"/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/freefield/wav/{x}.wav")
    # filepaths = train.filepaths.values.tolist()
    # for filepath in tqdm(filepaths, total=len(train)):
    #     fp = Path(filepath)
    #     save_path = Path(SAVE_PATH)
    #     save_path.mkdir(exist_ok=True, parents=True)
    #     save_file = str(save_path / f"{fp.stem}.npy")
    #     melspec = get_melspec(filepath, config)
    #     np.save(str(save_file), melspec)

    # CONFIG = "configs/melconfig2.yaml"
    # with open(CONFIG, "r") as fp:
    #     config = EasyDict(yaml.safe_load(fp))
    # DATA_ROOT = "data"
    # SAVE_PATH = "data/melspecs2"

    # train = pd.read_csv(Path(DATA_ROOT) / "train_metadata.csv")
    # train["filepaths"] = train.apply(lambda x: f"data/train_short_audio/{x['primary_label']}/{x['filename']}", axis=1,)
    # filepaths = train.filepaths.values.tolist()
    # folders = train.primary_label.values.tolist()
    # for filepath, folder in tqdm(zip(filepaths, folders)):
    #     fp = Path(filepath)
    #     save_path = Path(SAVE_PATH) / folder
    #     save_path.mkdir(exist_ok=True, parents=True)
    #     save_file = str(save_path / f"{fp.stem}.npy")
    #     melspec = get_melspec(filepath, config)
    #     np.save(str(save_file), melspec)

    # SAVE_PATH = "data/noisespecs2"

    # train = pd.read_csv("/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/freefield/labels.txt")
    # train = train.loc[train.hasbird == 0]
    # train["filepaths"] = train.itemid.apply(lambda x: f"/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/freefield/wav/{x}.wav")
    # filepaths = train.filepaths.values.tolist()
    # for filepath in tqdm(filepaths, total=len(train)):
    #     fp = Path(filepath)
    #     save_path = Path(SAVE_PATH)
    #     save_path.mkdir(exist_ok=True, parents=True)
    #     save_file = str(save_path / f"{fp.stem}.npy")
    #     melspec = get_melspec(filepath, config)
    #     np.save(str(save_file), melspec)

    SAVE_PATH = "data/noisespecs3"

    train = pd.read_csv("/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/birdvox_dcase_20k/labels.txt")
    train = train.loc[train.hasbird == 0]
    train["filepaths"] = train.itemid.apply(lambda x: f"/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/birdvox_dcase_20k/wav/{x}.wav")
    filepaths = train.filepaths.values.tolist()
    for filepath in tqdm(filepaths, total=len(train)):
        fp = Path(filepath)
        save_path = Path(SAVE_PATH)
        save_path.mkdir(exist_ok=True, parents=True)
        save_file = str(save_path / f"{fp.stem}.npy")
        melspec = get_melspec(filepath, config)
        np.save(str(save_file), melspec)