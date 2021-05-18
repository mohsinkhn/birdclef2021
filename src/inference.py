from collections import Counter
from itertools import chain

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from src.dataset import mono_to_color, INT2CODE, CODE2INT
from src.prepare_melspecs import get_melspec
from torchlib.io import ConfigParser


BASE_TEST_DIR = "data/train_soundscapes"


def read_data(filepaths, csvfile):
    test_info = pd.read_csv(csvfile)
    filenames = list(Path(filepaths).glob("*.ogg"))
    filename_map = {"_".join(f.stem.split("_")[:2]): str(f) for f in filenames}
    test_info["filepaths"] = test_info.row_id.apply(lambda x: filename_map["_".join(x.split("_")[:2])])
    # Looking for all unique audio recordings
    unique_audio_id = test_info.filepaths.unique()
    return test_info, unique_audio_id


def get_lat_lon(filepath):
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        lat = float(lines[3].split(" ")[1])
        lon = float(lines[4].split(" ")[1])
    return (lat, lon)


def get_locationwise_counts(location_filepaths, metadata_fp, normalize=True, range_deg=1):
    files = list(Path(location_filepaths).glob("*.txt"))
    file_locs = [f.stem.split("_")[0] for f in files]
    lat_lons = [get_lat_lon(fs) for fs in files]
    loc_coords = {loc: coord for loc, coord in zip(file_locs, lat_lons)}
    metadata = pd.read_csv(metadata_fp)
    bird_loc_cnts = {}
    for loc in file_locs:
        lat, lon = loc_coords[loc]
        print(lat, lon)
        subset = metadata.loc[(metadata.latitude.between(lat-range_deg, lat+range_deg) & 
                                          metadata.longitude.between(lon-range_deg, lon+range_deg))]
        subset['secondary_labels'] = subset['secondary_labels'].apply(eval)
        bird_cnts = Counter(subset.primary_label.tolist() + list(chain.from_iterable(subset.secondary_labels.tolist())))
        if normalize:
            bsum = sum(bird_cnts.values())
            bird_cnts = {b: c/bsum for b, c in bird_cnts.items()}
        bird_loc_cnts[loc] = bird_cnts
    return bird_loc_cnts


def get_audio_file_predictions(model, config, audio_file, test_audio, power=0.85, compress_factor=0.95, device='cuda'):
    melspectr = get_melspec(audio_file, config)
    melspectr = librosa.power_to_db(melspectr, amin=1e-7, ref=np.max)
    melspectr = ((melspectr+80)/80).astype(np.float32)
    site = test_audio.site.iloc[0]
    row_ids = test_audio.row_id.values
    end_seconds = test_audio.seconds.values
    clip_frames = int(config.sr/config.hop_length)
    ys = []

    # stack array of melspec images
    for s in end_seconds:
        image = melspectr[:, (s-5)*clip_frames:s*clip_frames]
        image = image/(image.max()+0.0000001)
        image = image**power
        image = mono_to_color(image, config.n_mels, int(compress_factor*config.width))
        ys.append(image)
    ys = np.stack(ys)

    # Make prediction
    batch_size = config.training.stage1.batch_size
    probas = []
    for n in range(0, len(ys), batch_size):
        if len(ys) == 1:
            mel = np.array(ys)
        else:
            mel = ys[n:n+batch_size]

        mel = torch.from_numpy(mel).to(device)
        prediction = model.model(mel)
        # prediction = torch.sigmoid(prediction)
        proba = prediction.detach().cpu().numpy()
        probas.append(proba)
    return row_ids, probas, site


def get_clipwise_preds(model, config, audio_files, test_df, power=0.85, compress_factor=0.95, device='cuda'):
    model.eval()
    row_idss, probass, sites = [], [], []
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            test_audio = test_df.query(f"filepaths == '{audio_file}'").reset_index(drop=True)
            row_ids, probas, site = get_audio_file_predictions(model, config, audio_file, test_audio, power,
                                                               compress_factor, device)                
            row_idss.extend(row_ids)
            probass.extend(probas)
            sites.append(site)
    return np.array(row_idss), np.concatenate(probass), np.array(sites)


def post_process2(probas, test, bird_loc_cnts, max_w=0.5, mean_w=0.5, cnt_w=0):
    proba = probas.copy()    
    for audio_id in test.audio_id.unique():
        rows = test.audio_id == audio_id
        site = test.loc[rows, 'site'].iloc[0]
        bird_cnts = bird_loc_cnts[site]
        cnt_prob = np.log1p(np.repeat(np.array([bird_cnts.get(INT2CODE[i], 0) for i in range(398)]).reshape(1, -1), sum(rows), 0))
        max_probs = np.max(proba[rows], axis=0, keepdims=True)
        mean_probs = np.mean(proba[rows], axis=0, keepdims=True)
        proba[rows] = proba[rows] + max_w*np.clip(max_probs, 0, 1.0) + mean_w*np.clip(mean_probs, 0, 1.0) + cnt_w*np.clip(cnt_prob, 0, 1/(1e-6 + cnt_w))
    return proba


def fast_f1_score(predictions, target):
    tp = (predictions * target).sum(1)
    fp = (predictions * (1 - target)).sum(1)
    fn = ((1 - predictions) * target).sum(1)
    f1 = tp / (tp + (fp + fn) / 2)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return f1.mean(), precision.mean(), recall.mean()


def get_score(probas, labels, thresh=0.5):
    targets = np.zeros_like(probas)
    for i, lb in enumerate(labels):
        for lj in lb.split(' '):
            targets[i, CODE2INT[lj]] = 1

    predictions = probas > thresh
    predictions[predictions.sum(1) == 0, 397] = 1 
    return fast_f1_score(predictions, targets)


def get_models_preds(models, configs, audio_files, test_df, power=0.85, compress_factor=0.95, device='cuda'):
    probass = []
    for model, config in zip(models, configs):
        row_ids, probas, sites = get_clipwise_preds(model, config, audio_files, test_df, power=0.85,
                                                    compress_factor=0.95, device='cuda')
        probass.append(probas)
    return row_ids, probass, sites


def get_pp_predictions(models, configs, audio_files, test_df, loc_files,
                       loc_csv, power=0.85, compress_factor=0.95, device='cuda',
                       max_w=0.5, mean_w=0.5, cnt_w=0.0, range_deg=0.2):
    row_ids, probs, sites = get_models_preds(models, configs, test_df.filepaths.unique(), test_df, power,
                                             compress_factor, device)
    probs = np.mean(probs, 0)
    bird_loc_cnts = get_locationwise_counts(loc_files, loc_csv, normalize=True, range_deg=range_deg)
    probs = sigmoid(probs)
    probs = post_process2(probs, test_df, bird_loc_cnts, max_w, mean_w, cnt_w)
    return probs, row_ids


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def generate(models, border, config, filepaths, csvfile, check_accuracy=True, device='cuda', power=1.0,
             lower_thresh=0.15, upper_thresh=0.3):
    preds = []

    # Uploading a list of files for testing | Загружаем список файлов для тестирования
    test_info = pd.read_csv(csvfile)
    filenames = list(Path(filepaths).glob("*.ogg"))
    filename_map = {"_".join(f.stem.split("_")[:2]): str(f) for f in filenames}
    test_info["filepaths"] = test_info.row_id.apply(lambda x: filename_map["_".join(x.split("_")[:2])])
    # Looking for all unique audio recordings
    unique_audio_id = test_info.filepaths.unique()
    # from src.constants import CODE2INT
    # train = pd.read_csv("data/train_metadata.csv")
    # tcor = train.loc[(train.longitude.between(-85, -83) & train.latitude.between(9, 11))]
    # tssw = train.loc[(train.longitude.between(-77, -76) & train.latitude.between(42, 43))]
    # bird_dict = {'COR': set([CODE2INT[b] for b in tcor.primary_label.unique()]),
    #              'SSW': set([CODE2INT[b] for b in tssw.primary_label.unique()])}

    # Predict
    for model in models:
        model.eval()
    with torch.no_grad():
        for audio_id in tqdm(unique_audio_id):
            # Getting a spectrogram
            melspectr = get_melspec(audio_id, config)
            melspectr = librosa.power_to_db(melspectr, amin=1e-7, ref=np.max)
            melspectr = ((melspectr+80)/80).astype(np.float16)
            # melspectr = melspectr - 0.1*melspectr.mean(1, keepdims=True)
            # Looking for all the excerpts for this sound
            test_df_for_audio_id = test_info.query(f"filepaths == '{audio_id}'").reset_index(drop=True)
            est_bird = np.zeros((config.num_classes))
            probass = {}
            for index, row in test_df_for_audio_id.iterrows():
                # Getting the site, start time, and id | Получаем сайт, время начала и id
                start_time = row['seconds'] - 5
                row_id = row['row_id']
                site = row['site']
                mels = []
                probas = None
                # indices = list(bird_dict[site])
                # Cut out the desired piece | Вырезаем нужный кусок
                start_index = int(config.sr * start_time/config.hop_length)
                end_index = int(config.sr * row['seconds']/config.hop_length)
                y = melspectr[:, start_index:end_index]
                # print(y.shape)
                # cutting off the tail | отсекаю хвост
                # if (y.shape[1] % config.width):
                #    y = y[:,:-(y.shape[1]%448)]
                
                prob = []
                for i, model in enumerate(models):
                    mels = []
                    probas = None
                    # Split into several chunks with the duration config.width
                    ys = np.reshape(y, (config.n_mels, -1, config.width))
                    ys = np.moveaxis(ys, 1, 0)
                    
                    # For each piece we make transformations | Для каждого куска делаем преобразования
                    for image in ys:
                        # Convert to 3 colors and normalize | Переводим в 3 цвета и нормализуем
                        # image = image - image.min()
                        image = image/(image.max()+0.0000001)
                        image = image**power

                        # image = image**0.85
                        # image = torch.from_numpy(np.stack([image, image, image])).float()
                        image = mono_to_color(image, config.n_mels, int(0.95*config.width))
                        mels.append(image)

                    mels = np.stack(mels)

                    batch_size = config.training.stage1.batch_size
                    for n in range(0, len(mels), batch_size):
                        if len(mels) == 1:
                            mel = np.array(mels)
                        else:
                            mel = mels[n:n+batch_size]

                        mel = torch.from_numpy(mel).to(device)

                        # Predict
                        prediction = model.model(mel)
                        # print(prediction)
                        prediction = torch.sigmoid(prediction)
                        # in numpy
                        proba = prediction.detach().cpu().numpy()

                        # Add zeros up to 265 | Добавить нули до 265
                        # proba = np.concatenate((proba, np.zeros((proba.shape[0], 265-proba.shape[1]))), axis=1)

                        # Adding to the array | Добавляю в массив
                        if probas is not None:
                            probas = np.append(probas, proba, axis=0)
                        else:
                            probas = proba
                    prob.append(probas)
                    # print(prob)
                # Averaging the ensemble | Усредняю ансамбль
                prob = np.stack(prob, axis=0)
                prob = prob**2
                proba = prob.mean(axis=0)  # gmean(prob)/2 + prob.mean(axis=0)/2
                proba = proba**(1/2)
                # print(proba)
                # If a bird is encountered in one segment, increase its probability in others
                # Если встретилась птица в одном отрезке, увеличить её вероятность в других
                for xx in proba:
                    z = xx.copy()
                    z[z < 0.5] = 0
                    est_bird = est_bird + z/70
                    est_bird[(est_bird < lower_thresh) & (est_bird > 0)] = lower_thresh
                    # est_bird[indices] += 0.01
                # Dictionary with an array of all passages | Словарь с массивом всех отрывков
                probass[row_id] = proba

            est_bird[est_bird > upper_thresh] = upper_thresh
            for row_id, probas in probass.items():
                prediction_dict = []
                for proba in probas:
                    proba += est_bird
                    # proba[indices] += 0.05
                    events = proba > border
                    labels = np.argwhere(events).reshape(-1).tolist()

                    # To convert in the name of the bird | Преобразовать в название птиц
                    if len(labels) == 0 or (397 in labels):
                        continue
                    else:
                        labels_str_list = list(map(lambda x: INT2CODE[x], labels))
                        for i in labels_str_list:
                            if i not in prediction_dict:
                                prediction_dict.append(i)

                # If birds are not predicted | Если не предсказываются птицы
                if len(prediction_dict) == 0:
                    prediction_dict = "nocall"
                else:
                    prediction_dict = " ".join(prediction_dict)

                # To add to the list | Добавить в список
                preds.append([row_id, prediction_dict])

        # Convert to DataFrame and save | Перевести в DataFrame и сохранить
        preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
        preds.to_csv('submission.csv', index=False)
        print(preds.head(10))
        if check_accuracy:
            actual = test_info[['row_id', 'birds']]
            actual.rename(columns={'birds': 'birdsa'}, inplace=True)
            checker = pd.merge(preds, actual, on='row_id', how='left')
            print(checker.tail())
            f1s, precs, recs = [], [], []
            eps = 1e-7
            for bp, ba in zip(checker.birds.values, checker.birdsa.values):
                true = set(bp.split(' '))
                pred = set(ba.split(' '))
                prec = len(true & pred) / (eps + len(pred))
                rec = len(true & pred) / (eps + len(true))
                f1 = 2 * prec * rec / (eps + prec + rec)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)
            f1score, precision, recall = np.mean(f1s), np.mean(precs), np.mean(recs)
            print(f"F1 - score : {f1score}, Precision: {precision}, Recall: {recall}")
            return preds, f1score
    return preds, None
