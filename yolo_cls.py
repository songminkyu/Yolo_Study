from ultralytics import YOLO
import cv2
# Load a model
model = YOLO("models/Yolo8/yolov8x-cls.pt")  # load an official model

result = model("@sample/weaponse/knife4.JPG")
plots = result[0].plot()
resize_img = cv2.resize(plots, (640, 640))
cv2.imshow("plot", resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()