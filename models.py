def get_model(model_name):
    if model_name == "resnet18":
        from torchvision.models import resnet18
        return resnet18(num_classes=10)
    elif model_name == "resnet34":
        from torchvision.models import resnet34
        return resnet34(num_classes=10)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        return resnet50(num_classes=10)
    elif model_name == "resnet101":
        from torchvision.models import resnet101
        return resnet101(num_classes=10)
    elif model_name == "resnet152":
        from torchvision.models import resnet152
        return resnet152(num_classes=10)
    elif model_name == "efficientnet-b7":
        # Load using HF
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained('google/efficientnet-b7')
    elif "vit" in model_name:
        from transformers import ViTForImageClassification
        return ViTForImageClassification.from_pretrained('google/%s' % model_name)
