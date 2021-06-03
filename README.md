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
├── generateCaption.py   
├── model  
│   ├── Attention.py  
│   ├── Seq2Seq.py  
│   └── local_constructor.py  
├── pytorch-i3d  
├── test.py  
├── train.py   

* **MSR-VTT** is the folder of dataset 
*  ` config.py ` is the onfigureation file 
*  **pytorch-i3d** is the pretrained feature extractors by the [i3d repository](https://github.com/piergiaj/pytorch-i3d/)
