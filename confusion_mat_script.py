import supervision as sv
import torch
from ultralytics import YOLO
import numpy as np

dataset = sv.DetectionDataset.from_yolo(images_directory_path="D:\Machine Learning\VisualAssist Data\VisualAssist.v1i.yolov5pytorch\train\images",annotations_directory_path="D:\Machine Learning\VisualAssist Data\VisualAssist.v1i.yolov5pytorch\train\labels",data_yaml_path="D:\Machine Learning\VisualAssist Data\VisualAssist.v1i.yolov5pytorch\data.yaml")

model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

confusion_matrix = sv.ConfusionMatrix.benchmark(
   dataset = dataset,
   callback = callback
)

confusion_matrix.plot()