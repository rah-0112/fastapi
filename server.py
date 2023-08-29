from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import librosa
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def some():
    return {"data": "Hello server"}

@app.route("/", methods=["POST"])
def predict():
    # get file from POST request and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000)) # generate file name as a dummy random number
    #wav_filename = str(random.randint(0, 100000))
    audio_file.save(file_name)
    y, sr = librosa.load(file_name)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.resize(mel_spec_db, (128, 128))
    # we don't need the audio file any more - let's delete it!
    os.remove(file_name)
    print(mel_spec_db)
    return {"data": mel_spec_db}