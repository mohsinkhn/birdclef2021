import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from src.dataset import mono_to_color, INT2CODE
from src.prepare_melspecs import get_melspec
from torchlib.io import ConfigParser


BASE_TEST_DIR = "data/train_soundscapes"


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
    return preds
