# import torchvision.models as models
# import torch
#
# resnet34=models.resnet34(pretrained=True)
# VGG=models.vgg16(pretrained=True)
# input=torch.randn((1,3,224,224))
# output=VGG(input)
# print(1)

import extract_features

model = extract_features.model_cnn_load()
file_name="/mnt/MSR-VTT/MSR-VTT/video0.mp4"
f = extract_features.extract_features(file_name, model)
