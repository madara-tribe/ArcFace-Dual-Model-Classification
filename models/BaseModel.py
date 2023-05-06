import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .layers import LambdaConv

model_name = "efficientnetv2_s"
class ArcFaceModel(nn.Module):
    def __init__(self, embedding_size, lambda_nn=True):
        super(ArcFaceModel, self).__init__()
        self.embedding_size = embedding_size
        self.lambda_nn = lambda_nn
        self.lambda_heads=4 
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.freeze_layer()
        self.replace()

    def freeze_layer(self):
        """
        print("{i} st child layer {child} is grad True")
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for i, child in enumerate(self.backbone.children()):
            if i > len(list(self.backbone.children()))-3:
                for param in child.parameters():
                    param.requires_grad = True

    def replace(self):
        if self.lambda_nn:
            num_feature=self.backbone.conv_head.out_channels
            self.backbone.bn2 = nn.Sequential(
                    LambdaConv(num_feature, num_feature, heads=self.lambda_heads, k=16, u=1),
                    nn.BatchNorm2d(num_feature),
                    nn.SiLU(inplace=True)
                    )
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, self.embedding_size)
    def forward(self, x):
        x = self.backbone(x)
        return x


class RecognitionModel(nn.Module):
    def __init__(self, color_size, shape_size, lambda_nn=True):
        super(RecognitionModel, self).__init__()
        self.color_size = color_size
        self.shape_size = shape_size
        self.lambda_nn = lambda_nn
        self.lambda_heads=4
        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.freeze_layer()
        self.replace()

    def freeze_layer(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for i, child in enumerate(self.backbone.children()):
            if i > len(list(self.backbone.children()))-3:
                print("{}st child layer {} is grad True".format(i, child))
                for param in child.parameters():
                    param.requires_grad = True

    def replace(self):
        num_feature=self.backbone.fc.in_features
        if self.lambda_nn:
            self.backbone.global_pool = nn.Sequential(
                    LambdaConv(num_feature, num_feature, heads=self.lambda_heads, k=16, u=1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1)
                    )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        self.color_fc = nn.Linear(512, self.color_size)
        self.shape_fc = nn.Linear(512, self.shape_size)

    def forward(self, x):
        x = self.backbone(x)
        c = self.color_fc(x)
        s = self.shape_fc(x)
        return c, s

