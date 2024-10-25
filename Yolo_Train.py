from ultralytics import YOLO

model = YOLO('models/Yolo8/yolov8x.pt')

results = model.train(data="@datasets/lvis.yaml", epoch=100, imgsz=640)