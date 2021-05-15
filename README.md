TODO:
30th April
* Setup baseline training and inference code in notebook
* Move code from notebook to config + runner
* Target baseline validation f1_score - 69%
* Setup debugging where appropriate

13th May - 
* Generate melspecs for faster iteration, 5 sec samples - 10 a.m 11th
* Setup SED model with densenet121, resnest50d, efficientnet-b0 - 12 a.m. 11th
* Use 5 second clips
* Add Gaussian, Pink, Band noise and BAD dataset as background noise - 5 p.m. 11th
* Random power for signal
* Use modified mixup - change mixup to use both labels - 6 p.m.
* Use secondary labels with 0.3 as target - 7 p.m.
* Setup evaluation --> if bird in segment with threshold greater than T, increase prior for that bird for complete file
* Submit on LB --> target 11 p.m. 12th May