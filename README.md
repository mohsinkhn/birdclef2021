TODO:
30th April
* Setup baseline training and inference code in notebook
* Move code from notebook to config + runner
* Target baseline validation f1_score - 69%
* Setup debugging where appropriate

13th May - 
* Generate melspecs for faster iteration, 5 sec samples
* Setup SED model with densenet121, resnest50d, efficientnet-b0
* Use 5 second clips
* Add Gaussian, Pink, Band noise and BAD dataset as background noise
* Random power for signal
* Use modified mixup - change mixup to use both labels
* Use secondary labels with 0.3 as target
* Setup evaluation --> if bird in segment with threshold greater than T, increase prior for that bird for complete file
* Submit on LB --> 

17th May
* effb0 baseline scores - 0.71
* res50_effb0+mix+rex different strtified folds - 0.71x
* Try simple kfold and average scores -- DONE - 0.73 for res26
* Restructure evaluation script, provision for location wise priors - Done
* res26d 5-kfold, effb0 5-kfold, res50 5-kfold - Done
* Post processing techniques - In progress
* Better evaluation metrics on sundscapes - Todo
* Add noise from soundscapes to training - Todo
* Find out echo augmentation - Todo
* Band pass filter for post processing, high frequency cutoff? - Todo
* log mean-max pooling from Jan Schutler - Todo  - P1
* combining channels with different weights - most files seem single channel - mostly mono files - not useful
* 30 second training - should be coupled with 0.5 weight for secondary labels
* Try pretraining on audioset? - Todo
* Training time improvements for faster iteration - Todo


19th May
* So far what has worked - used training augmentations from vlomme
* Combined res26 and effb0 preds
* Moving from vlomme post processing to logit model helped a bit, may be try GBM
* Backbone dicriminative experiments - 
* Different backbones - 
* External datasets
* Pretraining ?
* Adding nocall from current soundscapes as background noise, also changing background noises
* Restarts for current trained data
* More resolution ?
* Separate threshold for each site - Done

External datasets and pretraining seem to be strongest levers as of now. Current options for
external data are:
* additional birds recordings from xeno canto - could be used to pretrain for better finetuning
* soundscapes - add soundscapes from previous competitions and current one
* audioset pretraining - has helped in PANN based models