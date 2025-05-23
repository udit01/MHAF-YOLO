from ultralytics_mhaf import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('MAF-YOLOv2-n.yaml')
    model.train(data='coco.yaml', batch=16, device=0)
