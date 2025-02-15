from REID.logger.log import get_logger
import REID.config.model_cfgs as cfgs
from ultralytics import YOLO

log = get_logger(__name__)


class YoloDetect(object):
    def __init__(self, model_path=cfgs.YOLO_MODEL_PATH):
        # 验证模型路径是否为字符串
        if not isinstance(model_path, str):
            log.error("Model path must be a string.")
            raise ValueError("Model path must be a string.")
        try:
            # 创建模型
            self.model = YOLO(model_path)
            self.model_track = YOLO(model_path)
            log.info(f"{model_path} has already loaded")
        except FileNotFoundError:
            log.error(f"Model file not found at {model_path}.")
            raise
        except Exception as e:
            log.error(f"An error occurred while loading the model: {e}")
            raise

    # 使用YOLO模型进行探测, 并删去不符合的部分
    def detect(self, img, class_idx_list=cfgs.YOLO_LABELS, min_size=cfgs.YOLO_MIN_SIZE, conf=cfgs.YOLO_CONF,
               iou=cfgs.YOLO_IOU):
        boxes, clses = [], []
        results = self.model.predict(img, conf=conf, iou=iou, classes=class_idx_list)
        if results[0].boxes.xyxy is not None:
            _boxes = results[0].boxes.xyxy.cpu().tolist()
            _clses = results[0].boxes.cls.int().cpu().tolist()
            for box, cls in zip(_boxes, _clses):
                if box[3] - box[1] > min_size and box[2] - box[0] > min_size:
                    boxes.append(box)
                    clses.append(cls)
        return boxes, clses

    def track(self, frame, class_idx_list=cfgs.YOLO_LABELS, min_size=cfgs.YOLO_MIN_SIZE, conf=cfgs.YOLO_CONF,
              iou=cfgs.YOLO_IOU, persist=True, tracker=cfgs.YOLO_TRACKER_TYPE):
        boxes, track_ids, clses = [], [], []
        results = self.model.track(frame, persist=persist, tracker=tracker, conf=conf, iou=iou, classes=class_idx_list)
        if results[0].boxes.id is not None:
            _boxes = results[0].boxes.xyxy.cpu().tolist()
            _track_ids = results[0].boxes.id.int().cpu().tolist()
            _clses = results[0].boxes.cls.int().cpu().tolist()
            for box, track_id, cls in zip(_boxes, _track_ids, _clses):
                if box[3] - box[1] > min_size and box[2] - box[0] > min_size:
                    boxes.append(box)
                    clses.append(cls)
                    track_ids.append(track_id)
        return boxes, track_ids, clses

    def reset_track(self):
        self._model_track = YOLO(cfgs.YOLO_MODEL_PATH)
