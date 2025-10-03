import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalization values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def fix_extensions(root_dir):
    """
    Renames all .JPEG files to .jpg so torchvision ImageFolder can read them.
    """
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".JPEG"):
                old_path = os.path.join(subdir, f)
                new_path = os.path.join(subdir, f[:-5] + ".jpg")
                os.rename(old_path, new_path)
    print(f"Fixed extensions in {root_dir}")

def organize_val_images(val_dir):
    """
    Reorganizes Tiny ImageNet validation set into class-specific subfolders
    (only run once!)
    """
    img_dir = os.path.join(val_dir, 'images')
    ann_file = os.path.join(val_dir, 'val_annotations.txt')

    with open(ann_file, 'r') as f:
        data = f.readlines()

    val_img_dict = {}
    for line in data:
        parts = line.strip().split('\t')
        img_name, label = parts[0], parts[1]
        val_img_dict[img_name] = label

    for img, label in val_img_dict.items():
        label_dir = os.path.join(val_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        src = os.path.join(img_dir, img)
        dst = os.path.join(label_dir, img)
        if os.path.exists(src):
            shutil.move(src, dst)

    print("Validation set reorganized successfully!")

def get_dataloaders(train_dir, val_dir, batch_size=128, num_workers=4):
    fix_extensions("data/tiny-imagenet-200/train")
    fix_extensions("data/tiny-imagenet-200/val")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset.classes)

