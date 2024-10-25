import cv2
from ultralytics import YOLO, solutions

model = YOLO("models/yolov8m.pt")
names = model.model.names

cap = cv2.VideoCapture("@sample/SEAL.Team.S04E04.Shockwave.720p.AMZN.WEB-DL.DDP5.1.H.264-NTb.mkv")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

while cap.isOpened():
    try:
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)
        im0 = dist_obj.start_process(im0, tracks)
        video_writer.write(im0)
    except Exception as e:
        print("Error", e)

cap.release()
video_writer.release()
cv2.destroyAllWindows()