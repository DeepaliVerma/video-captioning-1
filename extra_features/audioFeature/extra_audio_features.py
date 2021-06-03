import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display as ld
import os

def AE(signal, frame_length):
    AE = []
    frame_size=frame_length
    num_frames = math.floor(signal.shape[0] / frame_length)
    for t in range(num_frames):
        lower = t * frame_size
        upper = (t + 1) * (frame_size - 1)
        AE.append(np.max(signal[lower:upper]))
    return np.array(AE)

if __name__=="__main__":
    audioPath=""
    for root, dir, files in os.walk(audioPath):
        for file in files:
            if file.endswith(".wav"):
                filePath=os.path.join(root,file)
                rb, sr = librosa.load(filePath)
                rap, _ = librosa.load(filePath)
                rock, _ = librosa.load(filePath)
                ZCRrb = librosa.feature.zero_crossing_rate(y=rb, frame_length=1024, hop_length=1024)
                ZCRrap = librosa.feature.zero_crossing_rate(y=rap, frame_length=1024, hop_length=1024)
                ZCRrock = librosa.feature.zero_crossing_rate(rock, frame_length=1024, hop_length=1024)
                fig, ax = plt.subplots(2, 3, figsize=(30, 15))
                ax[0, 0].set(title='Wave Form of R&B')
                ld.waveplot(rb, sr=sr, ax=ax[0, 0])
                ax[1, 0].set(title='ZCR Energy of R&B')
                ax[1, 0].plot(ZCRrb.T)
                ax[0, 1].set(title='Wave Form of Rap')
                ld.waveplot(rap, sr=sr, ax=ax[0, 1])
                ax[1, 1].set(title='ZCR Energy of Rap')
                ax[1, 1].plot(ZCRrap.T)
                ax[0, 2].set(title='Wave Form of Rock')
                ld.waveplot(rock, sr=sr, ax=ax[0, 2])
                ax[1, 2].set(title='ZCR Energy of Rock')
                ax[1, 2].plot(ZCRrock.T)

                ZCRrb = librosa.feature.zero_crossing_rate(y=rb, frame_length=1024, hop_length=1024)
                ZCRrap = librosa.feature.zero_crossing_rate(y=rap, frame_length=1024, hop_length=1024)
                ZCRrock = librosa.feature.zero_crossing_rate(rock, frame_length=1024, hop_length=1024)
                fig, ax = plt.subplots(2, 3, figsize=(30, 15))
                ax[0, 0].set(title='Wave Form of R&B')
                ld.waveplot(rb, sr=sr, ax=ax[0, 0])
                ax[1, 0].set(title='ZCR Energy of R&B')
                ax[1, 0].plot(ZCRrb.T)
                ax[0, 1].set(title='Wave Form of Rap')
                ld.waveplot(rap, sr=sr, ax=ax[0, 1])
                ax[1, 1].set(title='ZCR Energy of Rap')
                ax[1, 1].plot(ZCRrap.T)
                ax[0, 2].set(title='Wave Form of Rock')
                ld.waveplot(rock, sr=sr, ax=ax[0, 2])
                ax[1, 2].set(title='ZCR Energy of Rock')
                ax[1, 2].plot(ZCRrock.T)

