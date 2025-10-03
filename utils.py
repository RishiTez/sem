import os
import shutil
import csv
import glob

def prepare_tiny_imagenet(root_dir="data/tiny-imagenet-200"):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        images_path = os.path.join(class_path, "images")
        if os.path.isdir(images_path):
            jpg_files = glob.glob(os.path.join(images_path, "*.JPEG"))
            for f in jpg_files:
                new_f = os.path.join(class_path, os.path.basename(f)[:-5] + ".jpg")
                shutil.move(f, new_f)
            shutil.rmtree(images_path, ignore_errors=True)
            print(f"Flattened {class_name}: {len(jpg_files)} images moved.")

    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    if os.path.exists(val_images_dir) and os.path.exists(val_annotations_file):
        with open(val_annotations_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                img_name, label = row[0], row[1]
                label_dir = os.path.join(val_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                src = os.path.join(val_images_dir, img_name)
                dst = os.path.join(label_dir, img_name[:-5] + ".jpg")  
                if os.path.exists(src):
                    shutil.move(src, dst)
        shutil.rmtree(val_images_dir, ignore_errors=True)
        print("Validation images organized by class.")

    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    train_count = sum(len(glob.glob(os.path.join(train_dir, c, "*.jpg"))) for c in train_classes)
    val_count = sum(len(glob.glob(os.path.join(val_dir, c, "*.jpg"))) for c in val_classes)

    print("\n===== Tiny ImageNet Dataset Summary =====")
    print(f"Train classes: {len(train_classes)}")
    print(f"Val classes:   {len(val_classes)}")
    print(f"Train images:  {train_count}")
    print(f"Val images:    {val_count}")
    print("========================================\n")

