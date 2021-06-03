class Config():
    #Tokenizer
    numDecoderTokens=1500
    trainJson="./MSR-VTT/MSR-VTT-train.json" #"/mnt/MSR-VTT/MSR-VTT-train.json"
    valJson="./MSR-VTT/MSR-VTT-val.json" #"/mnt/MSR-VTT/MSR-VTT-val.json"
    totalJson="/mnt/MSR-VTT/MSR-VTT-total.json"

    shortestSentence=5
    longestSentence=12
    inputSentenceLength=10

    #Features
    featuresPath="/mnt/MSR-VTT/MSR-VTT-i3dfeatures/"

    #Encoder
    inputEncoderDims=1024
    outputEncoderDims=512

    #Decoder
    inputDecoderDims=1500
    outputDecoderDims=512

    #Dense
    tokenizerOutputdims=1500

    ##config for training
    BatchSize=128


    ##config for dataset
    extractFrames=80
    i3dModelPath="/root/Pycharm_Project/pytorch-i3d/model/rgb_imagenet.pt"

    ##config for final test
    checkpointPath='/root/Pycharm_Project/VideoCaption/final_checkpoint/best_model.pth'
    encoderPath="/root/Pycharm_Project/VideoCaption/final_checkpoint/encoder.pth"
    decoderPath="/root/Pycharm_Project/VideoCaption/final_checkpoint/decoder.pth"

myConfig=Config()
