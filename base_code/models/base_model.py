import torch.nn as nn
from torch import cat
from torchvision.models import resnet18


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
        # return x.squeeze()


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x


class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 2)  # 2 domains in the input (source and target)
        self.category_classifier = nn.Linear(512, 7)

        # The module receives a concatenation of the features extracted by the category encoder
        # and the features extracted by the domain encoder
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, branch):
        x = self.feature_extractor(x)
        if branch == -1:
            return x
        if branch == 0:  # category disentanglement, phase1
            x = self.category_encoder(x)  # disentangler D1
            x = self.category_classifier(x)
        elif branch == 1:  # category disentanglement, phase2
            x = self.category_encoder(x)  # disentangler D1
            x = self.domain_classifier(x)  # remember to freeze it in the main
        elif branch == 2:  # domain disentanglement, phase1
            x = self.domain_encoder(x)  # disentangler D2
            x = self.domain_classifier(x)
        elif branch == 3:  # domain disentanglement, phase2
            x = self.domain_encoder(x)  # disentangler D2
            x = self.category_classifier(x)  # remember to freeze it in the main
        else:  # reconstruction
            fcs = self.category_encoder(x)
            fds = self.domain_encoder(x)
            x = cat((fcs, fds), dim=1)  # to concatenate (maybe put the dimension that should be 0)
            x = self.reconstructor(x)
        return x


class ClipDisentangleModel(nn.Module):
    def __init__(self):
        super(ClipDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 2)  # 2 domains in the input (source and target)
        self.category_classifier = nn.Linear(512, 7)

        # The module receives a concatenation of the features extracted by the category encoder
        # and the features extracted by the domain encoder
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, branch):
        x = self.feature_extractor(x)
        if branch == -1:
            return x
        if branch == 0:  # category disentanglement, phase1
            x = self.category_encoder(x)  # disentangler D1
            x = self.category_classifier(x)
        elif branch == 1:  # category disentanglement, phase2
            x = self.category_encoder(x)  # disentangler D1
            x = self.domain_classifier(x)  # remember to freeze it in the main
        elif branch == 2:  # domain disentanglement, phase1
            x = self.domain_encoder(x)  # disentangler D2
            x = self.domain_classifier(x)
        elif branch == 3:  # domain disentanglement, phase2
            x = self.domain_encoder(x)  # disentangler D2
            x = self.category_classifier(x)  # remember to freeze it in the main
        elif branch == 4:
            x = self.domain_encoder(x)  # disentangler D2
        else:  # reconstruction
            fcs = self.category_encoder(x)
            fds = self.domain_encoder(x)
            x = cat((fcs, fds), dim=1)  # to concatenate (maybe put the dimension that should be 0)
            x = self.reconstructor(x)
        return x


# DG
class DomainGeneralizationModel(nn.Module):
    def __init__(self):
        super(DomainGeneralizationModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 3)  # 3 domains in the input (3 source domains)
        self.category_classifier = nn.Linear(512, 7)

        # The module receives a concatenation of the features extracted by the category encoder
        # and the features extracted by the domain encoder
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, branch):
        x = self.feature_extractor(x)
        if branch == -1:
            return x
        if branch == 0:  # category disentanglement, phase1
            x = self.category_encoder(x)  # disentangler D1
            x = self.category_classifier(x)
        elif branch == 1:  # category disentanglement, phase2
            x = self.category_encoder(x)  # disentangler D1
            x = self.domain_classifier(x)  # remember to freeze it in the main
        elif branch == 2:  # domain disentanglement, phase1
            x = self.domain_encoder(x)  # disentangler D2
            x = self.domain_classifier(x)
        elif branch == 3:  # domain disentanglement, phase2
            x = self.domain_encoder(x)  # disentangler D2
            x = self.category_classifier(x)  # remember to freeze it in the main
        else:  # reconstruction
            fcs = self.category_encoder(x)
            fds = self.domain_encoder(x)
            x = cat((fcs, fds), dim=1)  # to concatenate (maybe put the dimension that should be 0)
            x = self.reconstructor(x)
        return x

class ClipDisentangleModel_DomainGeneralization(nn.Module):
    def __init__(self):
        super(ClipDisentangleModel_DomainGeneralization, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 3)  # 3 domains in the input (3 source domains)
        self.category_classifier = nn.Linear(512, 7)

        # The module receives a concatenation of the features extracted by the category encoder
        # and the features extracted by the domain encoder
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, branch):
        x = self.feature_extractor(x)
        if branch == -1:
            return x
        if branch == 0:  # category disentanglement, phase1
            x = self.category_encoder(x)  # disentangler D1
            x = self.category_classifier(x)
        elif branch == 1:  # category disentanglement, phase2
            x = self.category_encoder(x)  # disentangler D1
            x = self.domain_classifier(x)  # remember to freeze it in the main
        elif branch == 2:  # domain disentanglement, phase1
            x = self.domain_encoder(x)  # disentangler D2
            x = self.domain_classifier(x)
        elif branch == 3:  # domain disentanglement, phase2
            x = self.domain_encoder(x)  # disentangler D2
            x = self.category_classifier(x)  # remember to freeze it in the main
        elif branch == 4:
            x = self.domain_encoder(x)  # disentangler D2
        else:  # reconstruction
            fcs = self.category_encoder(x)
            fds = self.domain_encoder(x)
            x = cat((fcs, fds), dim=1)  # to concatenate (maybe put the dimension that should be 0)
            x = self.reconstructor(x)
        return x
