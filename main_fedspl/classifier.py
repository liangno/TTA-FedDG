import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import random
import numpy as np
import numpy.random as npr
import math
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights

def conv1x1(input_channel, output_channel,bias=False):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=bias)


def conv3x3(in_channel, out_channel, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding,bias=bias)

def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_channel, output_channel, stride=1, downsample=None, track_running_stats=True):
        super(BasicBlock, self).__init__() 
        self.conv1 = conv3x3(input_channel, output_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False) 
        self.conv2 = conv3x3(output_channel, output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input 
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 
    def __init__(self, input_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = conv1x1(channel, channel*4)
        self.bn3 = nn.BatchNorm2d(channel*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input # skip path
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual

        return self.relu(out)

class Classifier(nn.Module):

    def __init__(self, args, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None

        # # 1) ResNet backbone (up to penultimate layer)
        # if not self.use_bottleneck:
        #     model = models.__dict__[args.arch](weights=ResNet18_Weights.IMAGENET1K_V1 if args.arch == 'resnet18' else ResNet50_Weights.IMAGENET1K_V1)
        #     modules = list(model.children())[:-1]
        #     self.encoder = nn.Sequential(*modules)
        #     self._output_dim = model.fc.in_features
            
        # # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        # else:
        #     model = models.__dict__[args.arch](weights=ResNet18_Weights.IMAGENET1K_V1 if args.arch == 'resnet18' else ResNet50_Weights.IMAGENET1K_V1)
        #     model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
        #     bn = nn.BatchNorm1d(args.bottleneck_dim)
        #     self.encoder = nn.Sequential(model, bn)
        #     self._output_dim = args.bottleneck_dim
        self.feature_norm = False
        self.backbone = False
        block=BasicBlock
        layers=[2,2,2,2]
        super(Classifier, self).__init__()
        self.input_channel = 64 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)      
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        self.fc_class = nn.Linear(512 * block.expansion, args.num_classes)
        print(args.num_classes)
        self.output_dim=512 * block.expansion
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc_class]
        for m in self.modules():            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channel, channel, stride=stride, downsample=downsample))
        self.input_channel = channel*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.input_channel, channel))
        return nn.Sequential(*layers)
        #self.fc_class = nn.Linear(self.output_dim, args.num_classes)


        ## Initialization and Masking 
        # for m in self.modules():
        #     # if isinstance(m, nn.Conv2d):
        #     #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal_(m.weight.data)   # Initializing with orthogonal rows

        # if self.use_weight_norm:
        #     self.fc_class = nn.utils.parametrizations.weight_norm(self.fc_class, dim=args.weight_norm_dim)



    # def forward(self, x, return_feats=False):
    #     # 1) encoder feature
    #     feat = self.encoder(x)
    #     feat = torch.flatten(feat, 1)

    #     logits = self.fc(feat)

    #     if return_feats:
    #         return feat, logits
    #     return logits
    def forward(self, x,return_feats=False):
        if x.size(-1) == 224:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
        elif x.size(-1) == 56:
            x = self.layer1(x)
            
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        feature_out = x
        self.backbone=True   
        if self.backbone:
            return  feature_out,self.fc_class(x)
        else:
            logit = self.fc_class(x)
            return logit

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.parameters())
        # case 2)
        else:
            #resnet = self.encoder[0]
        #     for module in list(self.children())[:-1]:
        #         backbone_params.extend(module.parameters())
        #     # bottleneck fc + (bn) + classifier fc
        #     # extra_params.extend(resnet.fc.parameters())
        #     # extra_params.extend(self.encoder[1].parameters())
        #     # extra_params.extend(self.fc.parameters())
        #     extra_params.extend(self.fc.parameters())
        # # Exclude frozen params
            for module in self.baselayer[:-1]:  # Exclude the last fc_class
                backbone_params.extend(module.parameters())
        
        # Add extra params
            extra_params.extend(self.fc_class.parameters())
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    #@property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0
