def get_model(model_name, dataset_name):
    # Determine num_classes based on dataset
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "tiny-imagenet":
        num_classes = 200

    # Vision models
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
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained('google/efficientnet-b7')
    elif "vit" in model_name:
        from transformers import ViTForImageClassification
        return ViTForImageClassification.from_pretrained('google/%s' % model_name)
    elif model_name == "densenet121":
        from torchvision.models import densenet121
        return densenet121(num_classes=num_classes)
    elif model_name == "densenet169":
        from torchvision.models import densenet169
        return densenet169(num_classes=num_classes)
    elif model_name == "densenet201":
        from torchvision.models import densenet201
        return densenet201(num_classes=num_classes)
    elif model_name == "densenet161":
        from torchvision.models import densenet161
        return densenet161(num_classes=num_classes)
    
    # NLP models
    elif model_name == "bert-base-uncased":
        from transformers import BertForSequenceClassification
        return BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        from transformers import GPT2LMHeadModel
        return GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in ["roberta-base", "roberta-large"]:
        from transformers import RobertaForSequenceClassification
        return RobertaForSequenceClassification.from_pretrained(model_name)
    elif model_name in ["distilbert-base-uncased", "distilbert-base-uncased-distilled-squad"]:
        from transformers import DistilBertForQuestionAnswering
        return DistilBertForQuestionAnswering.from_pretrained(model_name)
    elif model_name in ["albert-base-v1", "albert-large-v1"]:
        from transformers import AlbertForSequenceClassification
        return AlbertForSequenceClassification.from_pretrained(model_name)
    elif model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration.from_pretrained(model_name)
