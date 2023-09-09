from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import os
import numpy as np
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def predict(audio: UploadFile = File(...)):
    contents = await audio.read()
    with open(audio.filename, "wb") as f:
        f.write(contents)
    m4a_audio = AudioSegment.from_file(audio.filename, format="m4a")
    m4a_audio.export("audio.mp3", format="mp3")
    y, sr = librosa.load('audio.mp3')
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.resize(mel_spec_db, (128, 128))
    os.remove(audio.filename)
    os.remove('audio.mp3')
    res = list()
    for row in mel_spec_db:
        res.append(row.tolist())
    return {"data": res}