import os

import numpy as np
import soundfile as sf
import librosa
from datasets.audio_dataset import AudioDataset
from datasets.data_utils import create_noisy_clip_dir

input_dir = 'F:/datasets/libri_speech_subset'
data_path_noise = 'F:/datasets/Nonspeech'

output_dir = None

input_dir = os.path.split(input_dir + '/')[0]  # remove trailing / in case there is one
if output_dir is None:
    output_dir = input_dir + '_noisy'
if not os.path.exists(output_dir):  # create directory if it does not already exist
    os.mkdir(output_dir)
audio_files = os.listdir(input_dir)
audio_files_length = len(audio_files)


def soundfile_test():
    for i, audio_file in enumerate(audio_files):
        output_file_path = os.path.join(output_dir, audio_file)
        print('Writing file ' + str(i + 1) + '/' + str(audio_files_length) + ' ' + output_file_path)
        file_extension: str = os.path.splitext(audio_file)[1][1:]

        data, samplerate = sf.read(os.path.join(input_dir, audio_file))
        print(len(data))
        sf.write(output_file_path, data, samplerate)

def librosa_test():
    for i, audio_file in enumerate(audio_files):
        output_file_path = os.path.join(output_dir, audio_file)
        print('Writing file ' + str(i + 1) + '/' + str(audio_files_length) + ' ' + output_file_path)
        file_extension: str = os.path.splitext(audio_file)[1][1:]

        data, samplerate = librosa.load(path=os.path.join(input_dir, audio_file), sr=None)
        print(samplerate)
        sf.write(output_file_path, data, samplerate)


librosa_test()
dir_list = os.listdir(input_dir)
data, samplerate = sf.read(os.path.join(input_dir, dir_list[34]))
noise_list = os.listdir(data_path_noise)
noise, samplerate_noise = librosa.load(os.path.join(data_path_noise, noise_list[0]),sr=samplerate)

noise_clip = []
while len(noise_clip) < len(data):
    print(noise_clip)
    noise_clip = np.concatenate([noise_clip,noise])

noise_clip = noise_clip[:len(data)]
print(len(data))
print(len(noise_clip))
joined = data + noise_clip/2
joined /= 2
sf.write('audio/.lala.wav', joined , samplerate)
