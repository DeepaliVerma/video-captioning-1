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

# Testing: Caption Generation 
```
python generateCaption.py.py -videoFeature feature_path --inputType feature
```
