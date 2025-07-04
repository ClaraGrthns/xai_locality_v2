import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.dataset.cats_vs_dogs import CatsVsDogsDataset
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import train_test_split
from src.model.inception_v3 import InceptionV3BinaryClassifier

def get_config():
    return {
        'num_epochs': 60,
        'batch_size': 32,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'class_mapping_file': "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt",
        'train_path': "/common/datasets/ImageNet_ILSVRC2012/train",
        'specific_classes': ["Labrador retriever", "Persian cat"],
        'model_save_path': "/home/grotehans/xai_locality/pretrained_models/inception_v3/binary_cat_dog_best.pth",
        'log_dir': '/home/grotehans/xai_locality/train/runs'
    }

def setup_model(device):
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.AuxLogits.fc.parameters():
        param.requires_grad = True
        
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            
    model = model.to(device)
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_data(config):
    dataset = CatsVsDogsDataset("/home/grotehans/xai_locality/data/cats_vs_dogs/train", transform=get_transforms())

    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        random_state=42
    )
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for i, (images, labels, _) in enumerate(train_loader):
        labels = labels.to(dtype=torch.float)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            outputs, aux_outputs = outputs
            loss1 = criterion(outputs.view(-1), labels)
            loss2 = criterion(aux_outputs.view(-1), labels) * 0.3
            loss = loss1 + loss2
        else:
            loss = criterion(outputs.view(-1), labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs.view(-1)) > 0.5
            acc = (preds == labels).float().mean().item()
            running_accuracy += acc
            
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Accuracy', acc, epoch * len(train_loader) + i)
    
    return running_loss / len(train_loader), running_accuracy / len(train_loader)

def validate(model, val_loader, criterion, device, writer):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            labels = labels.to(dtype=torch.float)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss1 = criterion(outputs.view(-1), labels)
                loss2 = criterion(aux_outputs.view(-1), labels) * 0.3
                loss = loss1 + loss2
            else:
                loss = criterion(outputs.view(-1), labels)
                
            val_loss += loss.item()
            writer.add_scalar('Validation Loss', loss.item(), len(val_loader))
    
    return val_loss / len(val_loader)

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = setup_model(device)
    print("parameters unfreezed for training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
                list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters()), 
                lr=config['learning_rate']
            )    
    train_loader, val_loader = load_data(config)
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    best_val_loss = float('inf')
    best_model_wts = None
    
    for epoch in range(config['num_epochs']):
        epoch_loss, epoch_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        val_loss = validate(model, val_loader, criterion, device, writer)
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, config['model_save_path'])

    
    if best_model_wts is not None:
        torch.save(best_model_wts, config['model_save_path'])
        print("Best model saved!")
    
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()
