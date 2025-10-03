import torch.nn as nn
import torchvision.models as models

def get_teacher_model(num_classes=200):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # replace last layer
    return model
