from ultralytics import YOLO
import argparse

def test_and_evaluate(model_path, test_path):
    model = YOLO(model_path)

    print("Running inference on test set...")
    results = model.predict(source=test_path, save=True)

    print("Evaluation Metrics:")
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Evaluate YOLOv8 Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test images directory")
    
    args = parser.parse_args()
    test_and_evaluate(args.model_path, args.test_path)
