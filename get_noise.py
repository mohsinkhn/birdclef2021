import pandas as pd


train = pd.read_csv("/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/freefield/labels.txt")
train = train.loc[train.hasbird == 0]
train["numpyfilepaths"] = train.itemid.apply(lambda x: f"./data/noisespecs/{x}.npy")
train.to_csv("data/nocall.csv", index=False)

train = pd.read_csv("/home/mohsin_okcredit_in/projects/birdsong_identification/data/BAD/birdvox_dcase_20k/labels.txt")
train = train.loc[train.hasbird == 0]
train["numpyfilepaths"] = train.itemid.apply(lambda x: f"./data/noisespecs3/{x}.npy")
train.to_csv("data/nocall2.csv", index=False)
