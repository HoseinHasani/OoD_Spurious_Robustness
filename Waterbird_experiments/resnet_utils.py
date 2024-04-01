import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)  
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = F.adaptive_avg_pool2d(features4, (1, 1))
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        # Applying threshold to the output
        output = F.threshold(output, threshold, 0)
        return output

    def intermediate_forward(self, x, layer_index):
        out = self.model.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.maxpool(out)

        out = self.model.layer1(out)
        if layer_index == 1:
            return out

        out = self.model.layer2(out)
        if layer_index == 2:
            return out

        out = self.model.layer3(out)
        if layer_index == 3:
            return out

        out = self.model.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        # Applying threshold to the output
        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        # Choose the appropriate layer based on the layer_index
        if layer_index == 1:
            return self.model.layer1(x)
        elif layer_index == 2:
            return self.model.layer2(self.model.layer1(x))
        elif layer_index == 3:
            return self.model.layer3(self.model.layer2(self.model.layer1(x)))
        elif layer_index == 4:
            return self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        else:
            raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc

