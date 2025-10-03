import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.tiny_imagenet import get_dataloaders
from models.teacher import get_teacher_model
from utils import prepare_tiny_imagenet
import time

deivce = None

def validate(model, val_loader, criterion, device=None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f"Validation batch [{batch_idx}/{len(val_loader)}], Loss: {loss.item():.4f}")

    val_loss /= total
    acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc:.2f}%")
    return acc

def save_checkpoint(state, filename="checkpoint.pth.tar", folder="checkpoints"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device=None):
    if device is None: device = torch.device("cpu")
    model.to(device)
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    acc = 100.0 * correct / total
    print(f"Epoch [{epoch}] Training Loss: {epoch_loss:.4f}, Accuracy: {acc:.2f}%")
    return epoch_loss, acc

def main():
    data_root = "data/tiny-imagenet-200"
    train_dir = f"{data_root}/train"
    val_dir   = f"{data_root}/val"
    checkpoint_path = "checkpoints/teacher_best.pth.tar"

    if torch.backends.mps.is_available():  
        device = torch.device("mps")
    elif torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    prepare_tiny_imagenet(data_root)
    train_loader, val_loader, num_classes = get_dataloaders(train_dir, val_dir, batch_size=8)

    model = get_teacher_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}. Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = 0 
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")

    for epoch in range(start_epoch, 21):  
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, device=device)
        acc1 = validate(model, val_loader, criterion, device=device)
        scheduler.step()

        if acc1 > best_acc:
            best_acc = acc1
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, filename="teacher_best.pth.tar")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    time_taken = end_time - start_time
    hours_elapsed = int(time_taken // 3600)
    minutes_elapsed = int((time_taken % 3600) // 60)
    seconds_elapsed = int(time_taken % 60)
    print(f"Total time taken: {hours_elapsed}h {minutes_elapsed}m {seconds_elapsed}s")

