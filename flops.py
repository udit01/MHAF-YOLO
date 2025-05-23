from ultralytics_mhaf import YOLOv10

model = YOLOv10('MAF-YOLOv2-n.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()