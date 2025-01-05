from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('MAF-YOLOv2-n-obb.yaml')
    model.train(data='DOTAv1-noms.yaml', batch=12, device=0, epochs=200, mixup=0.1, scale=0.9, degrees=180, imgsz=1024)
