# Knowledge Distillation (Teacher-Student AI Model Compression)

## Table of Contents
1. [Introduction](#introduction)
2. [Concept: Knowledge Distillation](#concept-knowledge-distillation)
3. [Project Goal](#project-goal)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Conclusion](#conclusion)

---

## 1. Introduction

This project implements Knowledge Distillation (KD), a powerful model compression technique. The core idea is to train a smaller, computationally efficient **Student Model** to mimic the behavior and performance of a larger, pre-trained, and highly accurate **Teacher Model**.

This is crucial for deploying complex AI on resource-constrained environments like mobile devices or edge hardware where latency and memory are critical limitations.

---

## 2. Concept: Knowledge Distillation

In the Teacher-Student paradigm:
* The Teacher Model is first trained to a high accuracy on the original dataset.
* The Student Model is then trained using the **soft targets** (logits or probability distributions) from the Teacher, which contain richer information about class relationships than the hard (ground-truth) labels alone.
* A combined loss function is used: a Distillation Loss (to match the teacher's soft targets) and a **Student Loss** (to match the hard ground-truth labels).

Key Benefit: Achieve near-Teacher performance with significantly reduced model size and faster inference time.

---

## 3. Project Goal

The primary goals of this project are to:
1.  Train a high-performing Teacher Model.
2.  Implement the Knowledge Distillation training loop.
3.  Train a smaller Student Model.
4.  Compare the performance of the Student Model with and without distillation.

---

## 4. Installation

Dataset Used: Tiny ImageNet
Link: http://cs231n.stanford.edu/tiny-imagenet-200.zip

### Prerequisites
* Python 3.8+
* pip

### Setup
1. Download and extract dataset:
    ```bash
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip
    ```
2. Clone the repository:
    ```bash
    git clone https://github.com/RishiTez/sem.git
    cd sem
    git checkout VII
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

## 5. Usage

1. Perform Data preparation:
```bash
    python data_preparation.py 
```
2. Train the Teacher Model:
    ```bash
    python train_teacher.py 
    ```
    Training Time: 28Hrs 52Mins 47Secs

3. Train the Student Model without Teacher model for Baseline:
    ```bash
    python train_student.py --no-teacher
    ```
    Training Time: 7Hrs 49Mins 48Secs
    
4. Train the Student Model with Knowledge Distillation:
    ```bash
    python train_student.py
    ```
    Training Time: 14Hrs 45Mins 16Secs

Total Training Time: 2Days 3Hrs 27Mins 51Secs (On our device, Could vary based on hardware)

## Results

1. Teacher Model (ResNet-50)  
   - Final Accuracy: 57.81%  

2. Student Model (Baseline, no distillation)  
   - Final Accuracy: 54.55%  

3. Student Model (with Knowledge Distillation)  
   - Final Accuracy: 57.07%  

- The baseline student lags behind the teacher by ~3%.  
- With distillation, the student recovers most of this gap, achieving **+2.5% improvement** over the baseline.  
- The distilled student reaches near-teacher performance while being a lighter and more efficient model.  

---

### Conclusion
Knowledge distillation successfully improves the student model’s accuracy compared to naive training. While the distilled student did not surpass the teacher, it demonstrates that distillation can transfer knowledge effectively, making smaller models competitive without significant loss in accuracy.  

