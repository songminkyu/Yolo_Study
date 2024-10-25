from ultralytics import YOLO
import cv2
model = YOLO("models/deepfake_detected.pt") # 원하는 크기 모델 입력(n ~ x)

result = model.predict("@sample/bus.jpg", save=True, conf=0.5)
plots = result[0].plot()
resize_img = cv2.resize(plots, (640, 640))
cv2.imshow("plot", resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()