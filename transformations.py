from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

def get_transformation(model_name):
    if "vit" in model_name:
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    else:
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
