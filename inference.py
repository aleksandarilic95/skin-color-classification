import torch
import torchvision
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1

import yaml
import argparse
import cv2
from PIL import Image

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

mapping = {
    '0': 'White',
    '1': 'Black',
    '2': 'Latino Hispanic',
    '3': 'East Asian',
    '4': 'Indian',
    '5': 'Middle Eastern'
}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (0,0,255)
thickness              = 2
lineType               = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Inference a face detection Faster-RCNN network on WIDERFace dataset.")

    parser.add_argument('--config', type = str, help = 'Path to the training configuration file.', required = True)
    parser.add_argument('--config-detection', type = str, help = 'Path to the training configuration file.', required = True)
    parser.add_argument('--model-path', type = str, help = 'Path to the model file.', required = True)
    parser.add_argument('--model-detection-path', type = str, help = 'Path to the detection model file.', required = True)
    parser.add_argument('--image-path', type = str, help = 'Path to the image file.', required = True)
    
    opt = parser.parse_args()

    config = None
    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.Loader)

    config_detection = None
    with open(opt.config_detection, 'r') as f:
        config_detection = yaml.load(f, Loader = yaml.Loader)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_detection = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_model_detection = config_detection['MODEL']
    model_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained = cfg_model_detection['PRETRAINED'], 
        num_classes = cfg_model_detection['NUM_CLASSES']
    )

    model_detection.load_state_dict(torch.load(opt.model_detection_path, map_location = 'cpu'))

    model_detection.to(device)
    model_detection.eval() 

    cfg_model = config['MODEL']
    model = InceptionResnetV1(classify = True, num_classes = cfg_model['NUM_CLASSES'])

    model.load_state_dict(torch.load(opt.model_path, map_location = 'cpu'))

    model.to(device)
    model.eval()

    Image_PIL = Image.open(opt.image_path)
    Image_CV = cv2.imread(opt.image_path, cv2.COLOR_BGR2RGB)

    Image_tensor = data_transforms_detection(Image_PIL).unsqueeze(0).to(device)

    with torch.no_grad():
        output_detection = model_detection(Image_tensor)

    cfg_trainer_detection = config_detection['TRAINER']
    keep_indexes = torchvision.ops.nms(output_detection[0]['boxes'], output_detection[0]['scores'], cfg_trainer_detection['NMS_THRESHOLD'])
    
    detections = []
    for index in keep_indexes:
        if output_detection[0]['scores'][index] > cfg_trainer_detection['SCORE_THRESHOLD']:
            x1 = int(output_detection[0]['boxes'][index][0])
            y1 = int(output_detection[0]['boxes'][index][1])
            x2 = int(output_detection[0]['boxes'][index][2])
            y2 = int(output_detection[0]['boxes'][index][3])

            detections.append([x1, y1, x2, y2])   

    labels = []
    for detection in detections:
        Image_cropped = Image_CV[detection[1] : detection[3], detection[0] : detection[2], :]
        Image_tensor = data_transforms(Image_cropped).unsqueeze(0).to(device)

        output = model(Image_tensor)
        _, pred = torch.max(output, 1)

        label = mapping[str(pred.item())]
        labels.append({'bbox': detection,
                     'label': label})

    for label in labels:
        Image_CV = cv2.putText(
            Image_CV,
            label['label'], 
            (label['bbox'][0], label['bbox'][1] - 10), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType
        )

        Image_CV = cv2.rectangle(
            Image_CV,
            (label['bbox'][0], label['bbox'][1]),
            (label['bbox'][2], label['bbox'][3]),
            (0, 0, 255),
            1
        )
    
    Image_name = opt.image_path.split('/')[-1]
    cv2.imwrite('inference/inference_{}'.format(Image_name), Image_CV)