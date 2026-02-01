import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchvision.transforms import v2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Transforms
    val_transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Test Data
    test_dataset = datasets.ImageFolder(r"data\test", transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Load Model Architecture
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load Weights
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()

    # 4. Evaluation Loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    main()