import cv2
import os
import numpy as np
from torchvision import datasets, transforms
import torch

datasetTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( [0.485, 0.456, 0.406],  [0.229, 0.224, 0.225]),
])

def extractVideoFrames(videoPath,frameNums):
    videoData=None
    videoName=os.path.basename(videoPath)
    videoCapture = cv2.VideoCapture(videoPath)
    frameCount=int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = list(range(0, frameCount, frameCount // frameNums))
    sample = sample[:80]

    ret=True
    count=0
    while ret:
        ret,frame=videoCapture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if count in sample:
                frame = datasetTransform(frame)
                frame = torch.unsqueeze(frame, dim=0)
                if videoData is None:
                    videoData = frame
                else:
                    videoData = torch.cat((videoData, frame))

            count += 1

    videoData=torch.unsqueeze(torch.transpose(videoData,1,0),dim=0)
    return videoData

