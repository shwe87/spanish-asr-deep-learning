# Spanish ASR using Deep Learning
Spanish ASR created from scratch with deep learning, notebooks with results can be seen inside the folder **Results**.

The code has been modularized so the module files can be found under the folder utils:

* utils.py: This module has functions like plotting waveforms, spectrograms, saving files, models, etc.
* models.py: This is where all the models classes can be found.
* speechdataset.py: This module had to be used to pre-process the data before injecting them to the model.
* textprocessor.py: This module is used for pre-processing the utterances from the dataset like converting them to lower case, deleting the acccents, etc.

This was developed using kaggle and the data was used from:

* https://www.kaggle.com/carlfm01/120h-spanish-speech

* https://www.kaggle.com/bryanpark/spanish-single-speaker-speech-dataset

