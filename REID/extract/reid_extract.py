import numpy as np

import REID.config.model_cfgs as cfgs
from REID.logger.log import get_logger
import onnxruntime as ort
import torchvision.transforms.v2 as v2
import torch
import cv2
from PIL import Image
from IPython import embed
log = get_logger(__name__)


class ReIdExtract(object):
    def __init__(self, extract_class, onnx_model=cfgs.EXTRACTOR_PERSON, IN_SIZE=cfgs.REID_IN_SIZE,
                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.extract_class = extract_class
        self.onnx_model = onnx_model
        self.session = ort.InferenceSession(self.onnx_model, providers=providers)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.transform = v2.Compose([
            v2.Resize(IN_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        log.info(f"{onnx_model} has already loaded, the shape is [{self.input_width}, {self.input_height}]")

    def __call__(self, image_data, norm_feat=True):
        image_data = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        img = self.transform(image_data)
        input_var = torch.stack([img], dim=0)
        outputs = self.session.run(None, {self.model_inputs[0].name: input_var.numpy()})
        features = outputs[0][0]
        if norm_feat:
            features = features / (np.linalg.norm(features) + 1e-8)
        return features  # output image

if __name__ == "__main__":
    reid = ReIdExtract(cfgs.EXTRACTOR_PERSON)
    embed()