from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

#results = model(source="SEAL.Team.S04E04.Shockwave.720p.AMZN.WEB-DL.DDP5.1.H.264-NTb.mkv",show=True,conf=0.3, save=True)
results = model(source=0,show=True,conf=0.3, save=True)