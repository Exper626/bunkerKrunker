from ultralytics import YOLO
import os

MODEL_PATH = r"E:\bunkerKrunker\results_epoch_train1\yolov8s_25ep\weights\best.pt"
TEST_IMAGES = r"E:\bunkerKrunker\teaPick_data1\test\images"

SAVE_PROJECT = r"E:\bunkerKrunker\results_epoch_train1"
SAVE_NAME = "yolov8s_25ep_prediction"

# 🔹 Load model
model = YOLO(MODEL_PATH)

# 🔹 Run inference and save results
results = model.predict(
    source=TEST_IMAGES,
    save=True,              # saves images with predictions
    save_txt=True,          # saves YOLO-format detections in txt
    conf=0.25,              # confidence threshold
    project=SAVE_PROJECT,   # main results folder
    name=SAVE_NAME,         # subfolder
    exist_ok=True           # overwrite if already exists
)

print(f"✅ Inference complete. Results saved at: {os.path.join(SAVE_PROJECT, SAVE_NAME)}")