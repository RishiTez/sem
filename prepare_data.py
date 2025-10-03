from datasets.tiny_imagenet import organize_val_images

if __name__ == "__main__":
    val_dir = "data/tiny-imagenet-200/val"
    organize_val_images(val_dir)

