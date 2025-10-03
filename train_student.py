import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.tiny_imagenet import get_dataloaders
from models.teacher import get_teacher_model
from utils import prepare_tiny_imagenet
from train_teacher import save_checkpoint
from torchvision import models
import argparse

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce(student_logits, labels)

        p_s = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        p_t = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        loss_kl = self.kl_div(p_s, p_t) * (self.temperature ** 2)

        return self.alpha * loss_ce + (1 - self.alpha) * loss_kl

def train_one_epoch(student, teacher, loader, criterion, optimizer, device, epoch):
    student.train()
    if teacher is not None:
        teacher.eval()

    total, correct, running_loss = 0, 0, 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        student_outputs = student(inputs)

        if teacher is not None:
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            loss = criterion(student_outputs, teacher_outputs, targets)
        else:
            loss = criterion(student_outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    acc = 100.0 * correct / total
    print(f"Epoch [{epoch}] Training Loss: {epoch_loss:.4f}, Accuracy: {acc:.2f}%")
    return epoch_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    acc = 100.0 * correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {acc:.2f}%")
    return epoch_loss, acc

def main():
    parser = argparse.ArgumentParser(description="Train Student with/without Teacher Distillation")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Train student without teacher distillation (baseline).")
    args = parser.parse_args()

    use_teacher = not args.no_teacher

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("Mode:", "Distillation (Teacher)" if use_teacher else "Baseline (No Teacher)")

    data_root = "data/tiny-imagenet-200"
    train_dir = f"{data_root}/train"
    val_dir   = f"{data_root}/val"
    prepare_tiny_imagenet(data_root)

    train_loader, val_loader, num_classes = get_dataloaders(train_dir, val_dir)

    student = models.resnet18(weights=None, num_classes=num_classes).to(device)

    teacher = None
    if use_teacher:
        teacher = get_teacher_model(num_classes=num_classes).to(device)
        checkpoint = torch.load("checkpoints/teacher_best.pth.tar", map_location=device)
        teacher.load_state_dict(checkpoint["state_dict"])
        teacher.eval()
        print("Loaded teacher checkpoint for distillation.")

    if use_teacher:
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in range(1, 21):
        train_one_epoch(student, teacher, train_loader, criterion, optimizer, device, epoch)
        _, acc = validate(student, val_loader, nn.CrossEntropyLoss(), device)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            save_checkpoint({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "optimizer": optimizer.state_dict()
            }, filename="student_best.pth.tar")
            print("Student checkpoint saved.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    time_taken = end_time - start_time
    hours_elapsed = int(time_taken // 3600)
    minutes_elapsed = int((time_taken % 3600) // 60)
    seconds_elapsed = int(time_taken % 60)
    print(f"Total time taken: {hours_elapsed}h {minutes_elapsed}m {seconds_elapsed}s")
