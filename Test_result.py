import cv2
from ultralytics import YOLO

model = YOLO("runs1/train/yolo_tumor_bmshare_gpu/weights/best.pt")
results = model.predict("/home/alex/Desktop/Ml_project/Yolo_Project/brats/images/test/BraTS20_Training_369_127.png", conf=0.5, show=True)

for result in results:
    img = result.orig_img  # original image with detections drawn
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
