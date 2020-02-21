from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
from pydub.utils import mediainfo
import numpy as np

data_path = 'F:/datasets/libri_speech'

info = mediainfo("F:/datasets/Nonspeech/n29.wav")

song1 = AudioSegment.from_wav("F:/datasets/Nonspeech/n1.wav")
song2 = AudioSegment.from_wav("F:/datasets/Nonspeech/n29.wav")
song1 -= 10

new = AudioSegment.empty()
no_crossfade1 = song1.append(new, crossfade=0)
print(song1.duration_seconds)
print(song2.duration_seconds)
print(no_crossfade1.duration_seconds)
no_crossfade1.export("asfa.wav", format="wav")
song = song1.overlay(song2)
song.export("testlala.wav", format="wav")
song = song[:1000]
print(len(np.array(song.get_array_of_samples())))
print(len(song))
print(song.frame_count())
print(song.frame_rate)
print(song.frame_count() / song2.frame_rate)
print(song.duration_seconds)
plt.plot(song2.get_array_of_samples())
plt.show()



