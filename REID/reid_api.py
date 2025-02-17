from REID.detect.yolo_detector import YoloDetect
from REID.extract.reid_extract import ReIdExtract
from REID.logger.log import get_logger
import REID.config.model_cfgs as cfgs
from REID.search.search_engine import SearchEngine

log = get_logger(__name__)


# 单例模式
def Singleton(cls):
    _instance = {}

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance

    return wrapper


# 装饰器
@Singleton
class Reid(object):
    # 初始化
    def __init__(self, base_feat_lists, base_idx_lists, dims=1024, target_class="person", device_info="gpu"):
        self.target_class = target_class
        self.device_info = device_info
        self.detector = YoloDetect(cfgs.YOLO_MODEL_PATH)
        self.search_engine = SearchEngine(base_feat_lists, base_idx_lists, dims=dims)
        self.track_method = cfgs.YOLO_TRACKER_TYPE
        self.input_size = [256, 128]
        self.target_class_idx_list = [0]
        extractor_path = cfgs.EXTRACTOR_PERSON
        if "cpu" in self.device_info.lower():
            self.extractor = ReIdExtract(self.target_class, extractor_path, self.input_size,
                                         providers=['CPUExecutionProvider'])
        elif "gpu" in self.device_info.lower():
            self.extractor = ReIdExtract(self.target_class, extractor_path, self.input_size,
                                         providers=['CUDAExecutionProvider'])

    # 重载模型
    def reload_reid_model(self, in_extractor_path=None, target_class=None, device=None):
        if target_class is not None or device is not None:
            if target_class is not None:
                self.target_class = target_class
            else:
                self.device_info = device
            log.info(f"reload reid model, convert the model to {target_class} and based on {device}")
            self.input_size = [256, 128]
            self.target_class_idx_list = [0]
            extractor_path = cfgs.EXTRACTOR_PERSON
            if in_extractor_path is not None:
                extractor_path = in_extractor_path
            if "cpu" in self.device_info.lower():
                self.extractor = ReIdExtract(self.target_class, extractor_path, self.input_size,
                                              providers=['CPUExecutionProvider'])
            elif "gpu" in self.device_info.lower():
                self.extractor = ReIdExtract(self.target_class, extractor_path, self.input_size,
                                              providers=['CUDAExecutionProvider'])

    # 更新搜索引擎
    def reload_search_engine(self, base_feat_lists, base_idx_lists, dims=1024):
        log.info("reload faiss search engine")
        self.search_engine = SearchEngine(base_feat_lists, base_idx_lists, dims=dims)

    # 进行检测
    def detect(self, img, class_idx_list, input_format='image', is_track=True):
        if input_format == 'image':
            boxes, clses = self.detector.detect(img, class_idx_list=class_idx_list)
            track_ids = None
        elif input_format == 'video':
            if is_track:
                boxes, track_ids, clses = self.detector.track(img, class_idx_list, persist=True, tracker=self.track_method)
            else:
                boxes, clses = self.detector.detect(img, class_idx_list=class_idx_list)
                track_ids = None
        return boxes, track_ids, clses

    # 提取特征(归一化)
    def extract(self, img):
        each_img_norm_feat = self.extractor(img)
        return each_img_norm_feat

    # 搜索
    def search(self, img, bboxs, thresh=0.2):
        search_labels_list, search_dist_list = [], []
        before_sort_list = []
        filter_box_list = []
        for bbox in bboxs:
            each_crop_img = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
            each_img_norm_feat = self.extractor(each_crop_img)
            search_labels, search_dist = self.search_engine.search(each_img_norm_feat, 1)
            # 阈值
            if search_dist[0] <= thresh:
                search_labels_list.append(search_labels[0])
                search_dist_list.append(search_dist[0])
                filter_box_list.append(bbox)
                before_sort_list.append(search_labels[0])
            else:
                before_sort_list.append("unknown")
        return search_labels_list, search_dist_list, filter_box_list, before_sort_list

    # 重置
    def reset_track(self):
        self.detector.reset_track()