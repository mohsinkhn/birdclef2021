import librosa
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm
tqdm.pandas()

from src.constants import CODE2INT, TRAIN_CLIPS


def species_to_code(x):
    codes = []
    x = x[1:-1].split(",")
    if x[0] == "":
        return codes
    else:
        x = [code[1:-1] for code in x]
        return x


def get_timesteps(fname):
    #try:
    y, sr = librosa.load(fname, sr=None, res_type='kaiser_fast')
    return len(y)
    # except:
    #     raise ValueError(f"{fname} not found")
    # finally:
    #     return 0


if __name__ == "__main__":
    DATA_ROOT = "data"
    train = pd.read_csv(Path(DATA_ROOT) / "train_metadata.csv")
    train["filepaths"] = train.apply(
        lambda x: f"data/train_short_audio/{x['primary_label']}/{x['filename']}", axis=1,
    )
    train["labels"] = train.primary_label.apply(lambda x: [x])
    # train["durations"] = train.filepaths.progress_apply(get_timesteps)
    train["all_labels"] = train.secondary_labels.apply(species_to_code) + train["labels"]
    print(train.head())
    cvlist = list(KFold(5, shuffle=True, random_state=786).split(train))

    for idx, (tr_idx, vl_idx) in enumerate(cvlist):
        train.iloc[tr_idx][["filepaths", "labels", "all_labels"]].to_parquet(
            f"data/tr4_{idx}.pq"
        )
        train.iloc[vl_idx][["filepaths", "labels", "all_labels"]].to_parquet(
            f"data/vl4_{idx}.pq"
        )
