import torchvision
import torch
from torch import nn

class ResNet18(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.base_model = torchvision.models.resnet34(pretrained=True)
        #for i, param in enumerate(self.base_model.parameters()):
        #    param.requires_grad = False
        #    if i >= 15:
        #        break


        # Output resolution 15x9
        self.feature_extractor3 = nn.Sequential(
            #nn.ReLU(),
            #nn.BatchNorm2d(num_features = self.output_channels[2]),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= self.output_channels[2],
                out_channels= 512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= 512,
                out_channels= self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # Output resolution 8x5
        self.feature_extractor4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features = self.output_channels[3]),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= self.output_channels[3],
                out_channels= 1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 1024),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= 1024,
                out_channels= self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # Output resolution 4x3
        self.feature_extractor5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features = self.output_channels[4]),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= self.output_channels[4],
                out_channels= 512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
            #nn.Dropout(p = 0.1),
            nn.Conv2d(
                in_channels= 512,
                out_channels= self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            )
        )
        # Output resolution 2x1

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[4], 3, 3),
            shape(-1, output_channels[5], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        #out_features.append(x)
        x = self.base_model.layer2(x)
        out_features.append(x)
        x = self.base_model.layer3(x)
        out_features.append(x)
        x = self.base_model.layer4(x)
        out_features.append(x)
        x = self.feature_extractor3(x)
        out_features.append(x)
        x = self.feature_extractor4(x)
        out_features.append(x)
        x = self.feature_extractor5(x)
        out_features.append(x)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            out_channel = self.output_channels[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
