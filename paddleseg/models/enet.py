import paddle
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
import paddle.nn as nn
import paddle.nn.functional as F


@manager.MODELS.add_component
class ENet(nn.Layer):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self, backbone, num_classes, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
    def forward(self, x):
        features = self.backbone(x)
        return [features]
