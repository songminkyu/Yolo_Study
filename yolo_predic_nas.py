
#NAS 모델은 python 3.11 이상 되는 버전 지원 안함.
from ultralytics import NAS
import cv2
model = NAS("models/YoloNas/yolo_nas_l.pt") # 원하는 크기 모델 입력(n ~ x)

result = model.predict("@sample/drugs2.jpg", save=True, conf=0.5)
plots = result[0].plot()
resize_img = cv2.resize(plots, (640, 640))
cv2.imshow("plot", resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()