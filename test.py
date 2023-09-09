import librosa
from pydub import AudioSegment

m4a_audio = AudioSegment.from_file("audio.m4a", format="m4a")
m4a_audio.export("audio.mp3", format="mp3")
y, sr = librosa.load('audio.mp3')
print('load audio')