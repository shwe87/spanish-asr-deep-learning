# Please don't forget to mention that this was created by Shweta but please handle with care, if anything breaks it was not her responsibility.
import math
import random
import librosa
import string
import numpy as np
from random import choice
import pandas as pd
from torch.utils.data import Dataset
import soundfile as sf



# https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
def inject_noise(waveform):
    RMS = math.sqrt(np.mean(waveform**2))
    SNR = random.randint(-10, 35)
    denom = 10**(SNR/10)
    STD_n = math.sqrt(RMS**2/denom)
    noise =np.random.normal(0, STD_n, waveform.shape[0])
    waveform_with_noise = waveform + noise
    return waveform_with_noise


class SpanishSpeechDataSet(Dataset):
    """The audio should be pre-processed before feeding it to the NN."""
    def __init__(self, csv_files, root_dir, f_type='spec', num_samples=20000):
        self.root_dir = root_dir
        self.f_type = f_type
        # df1 has 11111 total data
        samples_df2 = num_samples - 11111
        # Create pandas dataframe as:
        #  Path | Label | Root_dir |
        df1 = pd.read_csv(csv_files[0],sep="|",header=None, names=["path","label"], usecols=[0,1])
        df1['root_dir'] = root_dir[0]
        # Only 20K rows
        df2 = pd.read_csv(csv_files[1], usecols=[0, 2], nrows=samples_df2)
        # Total of 31016 rows of data
        df2['root_dir'] = root_dir[1]
        df2.columns=['path','label', 'root_dir']        
        self.data = pd.concat([df1, df2], ignore_index=True)
        #Delete all those label rows with empty values:
        self.data.dropna(subset = ['label'], inplace=True)
        # Drop all containing digits:
        self.data = self.data.drop(self.data[self.data.label.str.contains(r'\d+')].index)
        # Expand the data frame to 
        # Path | Label | root_dir | dir | file | fullpath
        self.data[['dir','file']] = self.data.path.str.split("/",expand=True,)
        self.data['fullpath'] = self.data['root_dir'] + self.data['path']
        #To Lower case. We will only use the label one and ignore the utterance
        self.data['label'] = self.data['label'].str.lower()        
        # Remove punctuations:
        self.data['label'] = self.data['label'].str.replace('[{}]'.format(string.punctuation), '')
        # Convert to plain text. Remove accents and any other symbols like ¨, æ, etc. 
        self.data['label'] = self.data['label'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        # Now the text is cleaned to process.
        
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        INPUT:
        idx: The file path to the audio
        RETURNS:
        - waveform (Tensor): The waveform that the torchaudio returns.
        - sample rate (int): The sample rate that the torchaudio library returns.
        """
        # Load audio
        # Path | Label | root_dir | dir | file | fullpath
        full_path = self.data.iloc[idx]['fullpath']
        #The sampling rate that torchaudio uses for these audios are: 22050
        #sample_rate = 8000
        waveform, sample_rate = sf.read(full_path)
        #Randomly inject noise
        #inject = choice([0, 1])
        #if inject:
        #    waveform = inject_noise(waveform)       
        # Load text from pandas
        utterance = self.data.iloc[idx, 1]
        # Get the spectrogram if f_type = 'spec'
        if (self.f_type == 'spec'):
            # Calculate for every 23[ms] if sample_rate = 22050[Hz] and n_fft = 512 then in time = 23[ms]
            #signal = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=512)
            signal = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=512, fmin=10, fmax=8000, n_mels=64)
            #signal = librosa.util.normalize(signal)
        else:
            signal = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        #print("OK")
        # Return the spectrogram and the label
        #Scale the spectrograms:
        #scale(spectrogram)
        #print("Return Data Item")
        return (
            signal,
            utterance
        )
