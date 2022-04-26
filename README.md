# skin-color-classification

Skin color classification model trained on FairFace dataset.
Download [the dataset](https://drive.google.com/file/d/1RHxcydq9lI16lu4JFAkOYadvz2oNGj5x/view?usp=sharing) and place it in the folder data/fairface

To train the network, run:

    python3 train.py --config config/base.yaml
    
To test the network, run:

    python3 test.py --config config/base.yaml --model-path /path/to/model.pt
    
To inference a single image, run:

    python3 inference.py --config config/base.yaml --model-path /path/to/model.pt  --config-detection /path/to/face-detection/config/base.yaml --model-detection-path /path/to/detection/model.pt --image-path samples/test1.jpg
    
Inference results will be saved to inference/

Pretrained model (70% accuracy on validation set): [model20.pt](https://drive.google.com/file/d/1VIxYwrPcGvxIsvGpLsD3eHUoZW1Xydlw/view?usp=sharing)

Examples:
[test1.jpg](inference/inference_test1.jpg)
[test2.jpg](inference/inference_test2.jpg)