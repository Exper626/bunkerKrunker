from ultralytics import YOLO
import os

# paths
DATA_YAML = r"E:\bunkerKrunker\demeterGetsTea-3\data.yaml"
SAVE_DIR = r"E:\bunkerKrunker\results_dataset1_training1"

# array of epoch nos and models
epoch_list = [5, 10, 15, 20]
models = ["yolov8n.pt", "yolov8s.pt"]

# run the experiments
def run_experiments():
    for model_name in models:
        for epoch in epoch_list:
            print(f"Training {model_name} for {epoch} epochs...")
            model = YOLO(model_name)
            model.train(
                data = DATA_YAML,
                epochs = epoch,
                imgsz = 640,
                batch = 16,
                save_dir = SAVE_DIR,
                name = f"{model_name.split('.')[0]}_epoch_{epoch}"
            )
        print(f"Finished training {model_name} for {epoch} epochs.")

if __name__ == "__main__":
    run_experiments()