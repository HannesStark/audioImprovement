import os
from pydub import AudioSegment
from shutil import copy
import math
from sphfile import SPHFile


def split_from_libri_download() -> None:
    data_source = 'F:/datasets/train-clean-100.tar/LibriSpeech/train-clean-100'
    data_destination = 'F:/datasets/libri_speech'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_source):
        for file in f:
            if '.flac' in file:
                files.append(os.path.join(r, file))

    print(len(files))
    count = 0
    for f in files:
        count += 1
        print(count)
        copy(f, data_destination)
        print(f)

