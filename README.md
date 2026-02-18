## Project Report: Automated Car Damage Classification via Deep Learning

### 1. Project Overview

This project implements a robust supervised learning pipeline designed to automate the classification of car damage into six distinct categories. By architecting and evaluating multiple Convolutional Neural Network (CNN) configurations, the system provides a scalable solution for identifying damage patterns directly from digital imagery.

The primary objective is to deliver high-precision visual assessments that streamline workflows for insurance adjusters, auto repair technicians, and vehicle inspectors. By replacing manual inspection with automated detection, stakeholders can significantly accelerate claims processing and minimize human subjectivity in damage appraisal.

### 2. Core System Features

The technical architecture incorporates several engineering strategies to ensure high-performance inference and model reliability:

* Multiple CNN Architectures: Evaluation of a diverse model zoo ranging from lightweight custom layers to deep residual networks.
* Transfer Learning Implementation: Utilization of pre-trained ImageNet backbones to leverage complex feature hierarchies, significantly reducing training time.
* Data Augmentation Strategies: Integration of geometric and color-space transformations to mitigate overfitting and improve model generalization across varied lighting and angles.
* Hyperparameter Optimization: Systematic tuning of learning rates, dropout ratios, and batch sizes to ensure stable convergence and peak validation accuracy.
* Interactive Web Interface: A Streamlit-based deployment allowing for real-time model interaction and rapid prototyping.
* Comprehensive Evaluation Tools: Automated generation of confusion matrices and classification reports to provide granular insights into class-specific precision and recall.

### 3. Dataset and Image Preprocessing

**Data Composition**

The training environment is configured to ingest data from the ../dataset directory, where images are organized into six distinct damage classes (front breakage, front crushed, front normal, rear breakage, rear crushed, rear normal). The pipeline employs a 75%/25% training/validation split to maintain a rigorous evaluation holdout.

**Preprocessing Pipeline**

To ensure input consistency for the neural network backbones, the following transformations are enforced:

* **Resizing:** All input images are standardized to a resolution of 224x224 pixels to match the input requirements of the pre-trained backbones.
* **Normalization:** Pixel values are scaled using ImageNet statistics—mean: [0.485, 0.456, 0.406] and standard deviation: [0.229, 0.224, 0.225]—to ensure the input distribution aligns with the weights of the pre-trained models.

**Augmentation Techniques**

To expand the effective size of the training set and improve robustness against real-world image variability, implemented the following:

* Random horizontal flips.
* Random rotations to simulate varying camera orientations.
* Color jittering to account for different lighting conditions and sensor noise.

### 4. Model Architectures

The project followed an iterative architectural evolution, moving from baseline heuristics to sophisticated fine-tuned networks.

#### 4.1 Custom CNN

This 3-layer convolutional baseline (16, 32, 64 filters) served as the control. It utilizes ReLU activation and MaxPooling for spatial dimensionality reduction, with a dropout-heavy fully connected head to prevent the network from memorizing the training samples.

#### 4.2 CNN with Regularization

To address early-stage gradient instability, this iteration introduced Batch Normalization after each convolutional layer. Implemented L2 weight decay (1e-4) and increased the dropout rate to 0.5 to penalize large weights and prevent the model from becoming overly sensitive to specific training features.

#### 4.3 EfficientNet-B0 - Transfer Learning

This approach utilized a frozen feature extractor pre-trained on ImageNet. While computationally efficient, the frozen weights limited the model’s ability to distinguish subtle vehicle-specific damage patterns, as the backbone remained tuned for general object recognition rather than specialized automotive damage.

#### 4.4 ResNet50 (Optimal Model) - Transfer Learning

ResNet50 emerged as the superior architecture. Unlike the EfficientNet approach, we unfroze Layer4 and the fully connected layers for fine-tuning. This choice was deliberate: while earlier layers capture generic edges and textures, Layer4 captures high-level semantic features. Unfreezing these deeper layers allowed the model to adapt its complex shape-recognition capabilities to the specific nuances of vehicle damage, resulting in a significant performance delta over fully frozen models.

### 5. Performance Results and Optimization

**Key Metrics**

The fine-tuned ResNet50 architecture demonstrated the highest efficacy, balancing classification accuracy with manageable computational overhead.

**Metric	Result**
Validation Accuracy	~80%
(10 Epochs)

**Optimal Hyperparameters**

The following configuration, specifically optimized for the ResNet50 backbone, yielded the most stable convergence:

* Learning Rate: 0.0005
* Dropout: 0.3
* Batch Size: 32 (Selected for optimal gradient updates during fine-tuning)

### 6. Technology Stack

Our stack prioritizes performance, reproducibility, and ease of deployment.

* Deep Learning Frameworks: PyTorch, Pillow, torchvision.
* Pre-trained Models/Backbones: ResNet50, EfficientNet-B0.
* Data Science & Visualization: NumPy, Pandas, Matplotlib, scikit-learn.
* Deployment & Environment: Streamlit (Web Framework), Jupyter Notebook, Python 3.8+.

**Note:** Works best with images taken from the front or rear of the car.

To facilitate high-speed training and real-time inference on the web interface, the system supports full GPU acceleration.

### ⚙️ Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/Sugiuma/Deep_Learning_Project.git
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Run the streamlit app:
```bash
streamlit run app.py
```
### Experience the system in action: 
Explore the Live Inference Interface
