import torch
import os
import time
from ultralytics import YOLO
import argparse
import optuna
from optuna.samplers import TPESampler

# Function to train YOLOv8 with hyperparameters from Optuna
def train_yolo(data_yaml, model_size="yolov8n.pt", epochs=50, img_size=640, batch_size=16, learning_rate=0.01, momentum=0.937, weight_decay=0.0005):
    model = YOLO(model_size)  # Load YOLOv8 model
    
    start_time = time.time()
    
    # Train the model with the correct data.yaml path
    results = model.train(
        data=data_yaml,  # Use the correct path here
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,  # Initial learning rate
        momentum=momentum,   # Momentum for optimizer
        weight_decay=weight_decay,  # Regularization parameter
        lrf=0.01,  # Learning rate final value, for LR scheduler
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time, results

# Function to evaluate the model on the validation set
def evaluate_model(model, data_yaml):
    results = model.val(data=data_yaml)
    mAP = results.box.map50  # Mean Average Precision at IoU=0.5
    return mAP

# Objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("lr0", 1e-6, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    img_size = trial.suggest_categorical("img_size", [416, 640, 960])
    epochs = trial.suggest_int("epochs", 30, 100)
    momentum = trial.suggest_uniform("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    
    print(f"Starting trial with parameters: {trial.params}")
    
    # Use the correct data_yaml path passed from args
    model, training_time, results = train_yolo(
        data_yaml=args.data_yaml,  # Use the correct path here
        model_size="yolov8n.pt",
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Print loss values for each epoch
    for epoch, loss in enumerate(results.loss):
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
    
    # Evaluate the model and get the mAP score
    mAP = evaluate_model(model, args.data_yaml)
    
    print(f"Trial completed: mAP50 = {mAP:.4f}")
    
    return mAP  # The objective function returns mAP as the optimization target

# Function to optimize hyperparameters and train
def optimize_hyperparameters_and_train():
    # Create the Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize", sampler=TPESampler())  # Maximize mAP
    study.optimize(objective, n_trials=50, show_progress_bar=True)  # Display progress bar

    # Print best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    print(f"Best mAP: {study.best_value}")

    # Train final model with best parameters
    model, training_time, _ = train_yolo(
        data_yaml=args.data_yaml,  # Correct path is used here
        model_size="yolov8n.pt",
        epochs=best_params['epochs'],
        img_size=best_params['img_size'],
        batch_size=best_params['batch_size'],
        learning_rate=best_params['lr0'],
        momentum=best_params['momentum'],
        weight_decay=best_params['weight_decay']
    )

    print("Final model trained with the best hyperparameters.")

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test YOLOv8 for Shoplifting Detection with Hyperparameter Optimization")
    parser.add_argument("--mode", type=str, choices=["train", "test-image", "test-video", "hyperparameter-tuning"], required=True, help="Choose mode: train, test-image, test-video, or hyperparameter-tuning")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml for training")
    parser.add_argument("--model_size", type=str, default="yolov8n.pt", help="YOLO model size (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")

    args = parser.parse_args()

    # Check for mode
    if args.mode == "train":
        train_yolo(args.data_yaml, args.model_size, args.epochs)
    elif args.mode == "hyperparameter-tuning":
        optimize_hyperparameters_and_train()  # Optimizes and trains with best params
