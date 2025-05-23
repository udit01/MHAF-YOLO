from ultralytics_mhaf.models.yolo.detect import DetectionValidator
from ultralytics_mhaf.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        # return ops.non_max_suppression(
        #     preds["one2many"],
        #     self.args.conf,
        #     self.args.iou,
        #     labels=self.lb,
        #     multi_label=True,
        #     agnostic=self.args.single_cls,
        #     max_det=self.args.max_det,
        # )
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            return preds
        else:
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)