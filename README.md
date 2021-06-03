# Video Captioning 
Automatic description generation from digital video

## Feature Extraction 

```
cd ../extra_features
python extract_features.py
```

## Model Training 
```
python train.py
```

## Testing: Caption Generation 
```
python generateCaption.py.py -videoFeature feature_path --inputType feature
```
## Dataset
[MSR-VTT](https://www.mediafire.com/folder/h14iarbs62e7p/shared)

## File Description 
├── MSR-VTT  
│   ├── MSR-VTT  
│   ├── MSR-VTT-i3dfeatures  
│   ├── MSR-VTT-train.json 
│   └── MSR-VTT-val.json  
├── config.py
├── dataset
│   ├── ParserJson.py
│   └── VideoCaptionDataset.py
├── evaluation
│   └── coco_caption
├── extra_features
│   ├── 2D-CNN.py
│   ├── audioFeature
│   ├── config.py
│   ├── data
│   ├── extract_features.py
│   ├── flownet
│   ├── load_video.py
│   ├── model
│   └── pytorch_i3d.py
├── final_checkpoint
│   └── best_model.pth
├── frame.jpg
├── generateCaption.py
├── loss.txt
├── model
│   ├── Attention.py
│   ├── Seq2Seq.py
│   ├── __pycache__
│   └── local_constructor.py
├── pytorch-i3d
│   ├── LICENSE.txt
│   ├── README.md
│   ├── charades_dataset.py
│   ├── charades_dataset_full.py
│   ├── extract_features.py
│   ├── models
│   ├── pytorch_i3d.py
│   ├── train_i3d.py
│   └── videotransforms.py
├── test.py
├── train.py
└── train_results_loss.png
