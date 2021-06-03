import torch
from torch.utils.data import DataLoader
from config import myConfig
from model.Seq2Seq import Seq2SeqModel
from dataset.VideoCaptionDataset import VideoCaptionDataset
from tqdm import tqdm
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2SeqModel(config=myConfig)
model.to(device)

trainDataset = VideoCaptionDataset(config=myConfig, json=myConfig.trainJson)
trainDataloader = DataLoader(dataset=trainDataset, batch_size=myConfig.BatchSize, shuffle=True, drop_last=True)

testDataset = VideoCaptionDataset(config=myConfig, json=myConfig.valJson)
testDataloader = DataLoader(dataset=testDataset, batch_size=16, shuffle=False, drop_last=False)

train_size = len(trainDataloader)
test_size = len(testDataloader)

checkpoint = torch.load('/root/Pycharm_Project/VideoCaption/checkpoint_better/checkpoint_200.pth')
model.load_state_dict(checkpoint['net'])

model.eval()

total=0
correct=0
with torch.no_grad():
    print('====== Test Model ======')
    for data in tqdm(trainDataloader, leave=False, total=train_size):
        features, decoderInput, decoderTarget = data
        features = features.view(features.size()[0], features.size()[3], -1)
        features = features.to(device)
        decoderInput = decoderInput.to(device)
        decoderTarget = decoderTarget.to(device, dtype=torch.long)
        decoderTarget = decoderTarget.view(-1, )
        outputs,_ = model(features, decoderInput)
        outputs = outputs.contiguous().view(-1, myConfig.tokenizerOutputdims)
        _, predicted = outputs.max(1)
        total += decoderTarget.size(0)
        correct += predicted.eq(decoderTarget).sum().item()

    accuracy=correct/total*100
    print("====== Epoch {} accuracy is {}% ====== ".format(150,accuracy))

