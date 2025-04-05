from efficientnet_pytorch import EfficientNet
import torch.nn as nn
class EfficientNetFeatures(nn.Module):
    def __init__(self, version='efficientnet-b0'):
        super(EfficientNetFeatures, self).__init__()
        self.model = EfficientNet.from_pretrained(version)
        self.stages = nn.ModuleList(self.model._blocks)

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)

        x1 = None
        x2 = None
        x3 = None

        for idx, block in enumerate(self.stages):
            x = block(x)
            if idx == 2:  # early block
                x1 = x
            elif idx == 10:  # mid block
                x2 = x
            elif idx == 15:  # deep block
                x3 = x
                break
        
        # print('x1 shape:', x1.shape)  # After block[2]
        # print('x2 shape:', x2.shape)  # After block[10]
        # print('x3 shape:', x3.shape)  # After block[15]
        return x1, x2, x3