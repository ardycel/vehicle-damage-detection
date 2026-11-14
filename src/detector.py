from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

class DamageDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image: Image.Image, conf_threshold: float = 0.25):
        # PIL görüntüsünü OpenCV formatına (BGR) çevir
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Tahmin yap
        results = self.model.predict(source=img_bgr, conf=conf_threshold)
        result = results[0]

        # Kutucukların çizildiği görseli al ve RGB'ye geri çevir
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Tespit edilen nesnelerin detaylarını listele
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            detections.append({"label": label, "confidence": round(confidence * 100, 2)})

        return Image.fromarray(annotated_rgb), detections