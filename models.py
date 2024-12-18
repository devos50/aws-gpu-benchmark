def get_model(model_name, dataset_name):
    # Determine num_classes based on dataset
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "tiny-imagenet":
        num_classes = 200

    if model_name == "resnet18":
        from torchvision.models import resnet18
        return resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        from torchvision.models import resnet34
        return resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        return resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        from torchvision.models import resnet101
        return resnet101(num_classes=num_classes)
    elif model_name == "resnet152":
        from torchvision.models import resnet152
        return resnet152(num_classes=num_classes)
    elif model_name == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large
        return mobilenet_v3_large(num_classes=num_classes)
    elif model_name == "efficientnet-b7":
        # Load using HF
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained('google/efficientnet-b7')
    elif "vit" in model_name:
        from transformers import ViTForImageClassification
        return ViTForImageClassification.from_pretrained('google/%s' % model_name)
