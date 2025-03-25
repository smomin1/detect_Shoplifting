# detect_Shoplifting

## Project Description
This project aims to detect instances of shoplifting using YOLOv8, a state-of-the-art object detection model. The model is trained to recognize suspicious activities that indicate potential shoplifting behavior.

## YOLOv8 Model Variants
YOLOv8 comes in different sizes:
- **YOLOv8n (Nano)**: Best for edge devices and fast inference.
- **YOLOv8s (Small)**: A balance between speed and accuracy.
- **YOLOv8m (Medium)**: Suitable for general-purpose object detection.
- **YOLOv8l (Large)**: More accurate but computationally expensive.
- **YOLOv8x (Extra Large)**: Highest accuracy but requires significant resources.

For this project, **YOLOv8n** is used for efficiency.

## Install Required Libraries
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

## Install YOLOv8
To install YOLOv8, run:
```sh
pip install ultralytics
```

## Training the Model
To train the YOLOv8 model, use the following command:
```sh
python train_yolov8.py --mode train --data_yaml data.yaml --model_size yolov8n.pt --epochs 25
```
### Adjustable Parameters
- `--epochs`: Number of training epochs (default: 25 for efficiency).
- `--img_size`: Image size (default: 640).
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for optimization.
- `--momentum`: Momentum for optimizer.
- `--weight_decay`: Regularization parameter.

## Trained Model Location
After training, the model weights will be saved at:
```sh
./runs/detect/train/weights/best.pt
```

## Evaluating the Model
Once trained, evaluate the model using:
```sh
python eval_model.py --model_path ./runs/detect/train/weights/best.pt --test_path path/to/test/images
```

## How to Use This Model
This trained model can be integrated into a security surveillance system to:
- Detect potential shoplifting incidents in real time.
- Send alerts when suspicious activity is detected.
- Improve store security by reducing losses due to theft.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)

