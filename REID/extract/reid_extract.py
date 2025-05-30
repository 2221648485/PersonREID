import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
import REID.config.model_cfgs as cfgs
from REID.logger.log import get_logger

log_info = get_logger(__name__)


class ReIdExtract(object):
    def __init__(self, extract_class, onnx_model=cfgs.EXTRACTOR_PERSON, IN_SIZE=cfgs.REID_IN_SIZE,
                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self._extract_class = extract_class
        self._onnx_model = onnx_model
        # 使用onnx进行推理加快速度
        self.session = ort.InferenceSession(self._onnx_model, providers=providers)
        # 获取输入
        self.model_inputs = self.session.get_inputs()
        # 存储输入形状
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.transform = T.Compose([
            T.Resize(IN_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        log_info.info("{} model loaded!!! The shape is {}_{}.".format(onnx_model, self.input_width, self.input_height))

    # 推理
    def __call__(self, image_data, norm_feat=True):
        image_data = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        img = self.transform(image_data)
        input_var = torch.stack([img], dim=0)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_var.numpy()})
        features = outputs[0][0]
        if norm_feat:
            features = features / np.linalg.norm(features)
        return features  # output image


if __name__ == '__main__':
    reid_detector = ReIdExtract([0], "D:\\py\\PersonREID\\models\\reid_person_0.737.onnx")
    img = cv2.imread("../demo.jpg")
    reid_detector(img)
