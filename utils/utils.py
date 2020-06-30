# %% [code]
# Please don't forget to mention that this was created by Shweta but please handle with care, if anything breaks it was not her responsibility.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import os.path
from os import path

import matplotlib
import matplotlib.pyplot as plt
from datetime import date

import pickle as pkl
import librosa
import librosa.display

from torch import nn
import torch

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def plot_mfccs(m):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(m, x_axis='time')
    plt.colorbar()
    plt.title("13 MFCCs")
    plt.tight_layout()
    plt.show()

    
def plot_spec(sp, title=''):
    plt.figure(figsize=(10,5))
    sp_DB = librosa.power_to_db(sp, ref=np.max)
    librosa.display.specshow(sp_DB, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrograms of " + title)
    #plt.canvas.draw()
    #plt.tight_layout()
    plt.show()
    
def plot_raw_audio(audio_data, sr, title=''):
    """
    INPUTS:
    - audio_data: tensor of the raw audio.
    - sr: Sample Rate
    - label: Utterance of the audio.
    """
    plt.figure(figsize=(10,5))
    librosa.display.waveplot(audio_data, sr=sr)
    title = "Raw signal of the file " + title
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_all(audio_data, spec, sr, file, mfccs=None):
    plot_raw_audio(audio_data, sr, file)
    plot_spec(spec, file)
    #plot_mfccs(mfccs)
    
    
        
def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    text_transform = TextProcessing()
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int2text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int2text(decode))
    return decodes, targets


def save_checkpoint(save_path, model, optimizer, epoch, loss):
    today = date.today()
    torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss
                }, save_path+'_epoch_'+str(epoch)+'_'+str(today)+'.pth')

def load_checkpoint(load_path, optimizer):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return (model, optimizer, epoch, loss)

def save_model(save_path, model):
    save_filename = save_path + 'trained' + '.pt'
    torch.save(model.state_dict(), save_filename)


def load_model(load_path):
    save_filename = load_path + 'trained' + '.pt'
    return model.load_state_dict(torch.load(load_path, map_location=device))

def write_to_csv(file, decoded_pred, decoded_targets, test_loss, epoch=0):
    d = [{'Predicted': decoded_pred, 'Utterance':decoded_targets, 'test_loss': test_loss, 'epoch':epoch}]
    df = pd.DataFrame(d, columns=['Predicted', 'Utterance', 'test_loss','epoch'])
    if path.exists(file):
        df.to_csv(file, mode='a', header=False)
    else:
        df.to_csv(file, mode='a')
