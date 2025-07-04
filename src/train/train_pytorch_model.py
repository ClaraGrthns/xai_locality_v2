import sys
import os
import os.path as osp
sys.path.append(osp.join(os.getcwd(), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np

from src.model.pytorch_models_handler import LogReg, LinReg
from src.utils.pytorch_frame_utils import tensorframe_to_tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch model')
    parser.add_argument('--dataset', type=str, default="year")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print progress during training')
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna optimization trials')
    parser.add_argument('--data_path', type=str, default='/home/grotehans/xai_locality/data/LightGBM_year_normalized_data.pt', help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='/home/grotehans/xai_locality/pretrained_models/LinReg/year/LinReg_normalized_regression_year_results.pt', help='Path to save the trained model')
    parser.add_argument('--regression', action='store_true', help='Train a regression model instead of classification')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of Optuna optimization trials')
    return parser.parse_args()

def train_model(X, y, X_val, y_val, model, optimizer, criterion, epochs, weight_decay=0.0, verbose=False):
    # Training loop with validation
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X).flatten()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).flatten()
            val_loss = criterion(val_outputs, y_val).item()
            
            # Save best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    
    # Restore best model based on validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return best validation loss
    return model, best_val_loss


def objective_classification(trial, X, y, X_val, y_val, input_size, epochs, verbose):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True)
    
    model = LogReg(input_size=input_size, output_size=1)
    criterion = nn.BCELoss()
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    _, val_loss = train_model(X, y, X_val, y_val, model, optimizer, criterion, epochs, verbose=verbose)
    
    return val_loss


def objective_regression(trial, X, y, X_val, y_val, input_size, epochs, verbose):
    # Learning rate with wider range
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    
    # More optimizer options
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW", "RMSprop"])
    
    # Regularization strength
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True)
    
    # Try different momentum values for SGD
    momentum = trial.suggest_float("momentum", 0.0, 0.99) if optimizer_name == "SGD" else 0.0
    
    # Try different beta values for Adam-based optimizers
    beta1 = trial.suggest_float("beta1", 0.8, 0.99) if optimizer_name in ["Adam", "AdamW"] else 0.9
    beta2 = trial.suggest_float("beta2", 0.8, 0.999) if optimizer_name in ["Adam", "AdamW"] else 0.999
    
    # Try different batch sizes if using mini-batch training
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    
    model = LinReg(input_size=input_size)
    criterion = nn.MSELoss()
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    _, val_loss = train_model(X, y, X_val, y_val, model, optimizer, criterion, epochs, verbose=verbose)
    return val_loss


def main(args=None):
    if args is None:
        args = parse_args()
    print(args)

    is_regression = args.regression
    
    # Create TensorBoard logger
    model_type = "regression" if is_regression else "classification"
    log_dir = f"runs/{model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    data_path = args.data_path
    model_path = args.model_path
    if not osp.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    data = torch.load(data_path, weights_only=False)
    train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
    X = tensorframe_to_tensor(train_tensor_frame)  # All training features
    y = train_tensor_frame.y.float()  # All training labels as float
    X_val = tensorframe_to_tensor(val_tensor_frame)  # Validation features
    y_val = val_tensor_frame.y.float()  # Validation labels as float
    X_test = tensorframe_to_tensor(test_tensor_frame)
    y_test = test_tensor_frame.y.float()  # Test labels as float

    input_size = X.shape[1]
    
    if args.optimize:
        args.epochs = 100
        print("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction="minimize")
        
        if is_regression:
            study.optimize(
                lambda trial: objective_regression(trial, X, y, X_val, y_val, input_size, args.epochs, verbose=True), 
                n_trials=args.num_trials
            )
        else:
            study.optimize(
                lambda trial: objective_classification(trial, X, y, X_val, y_val, input_size, args.epochs, verbose=True), 
                n_trials=args.num_trials
            )
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        writer.add_hparams(best_params, {'hparam/best_loss': study.best_value})
        
        if is_regression:
            model = LinReg(input_size=input_size)
            criterion = nn.MSELoss()
        else:
            model = LogReg(input_size=input_size, output_size=1)
            criterion = nn.BCELoss()
        
        # Set up optimizer with best parameters
        if best_params["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        
        # Training with TensorBoard logging and validation
        best_val_loss = float('inf')
        for epoch in range(args.epochs*2):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X).flatten()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).flatten()
                val_loss = criterion(val_outputs, y_val)
                
                if is_regression:
                    val_preds = val_outputs.cpu().numpy()
                    val_r2 = r2_score(y_val.cpu().numpy(), val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val.cpu().numpy(), val_preds))
                    writer.add_scalar('Metrics/val_r2', val_r2, epoch)
                    writer.add_scalar('Metrics/val_rmse', val_rmse, epoch)
                else:
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                    val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
                    val_accuracy = ((val_probs > 0.5) == y_val.cpu().numpy()).mean()
                    writer.add_scalar('Metrics/val_auroc', val_auroc, epoch)
                    writer.add_scalar('Metrics/val_accuracy', val_accuracy, epoch)
                
                writer.add_scalar('Loss/validation', val_loss, epoch)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
            
            if (epoch + 1) % 10 == 0:
                if is_regression:
                    print(f'Epoch [{epoch+1}/{args.epochs*2}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{args.epochs*2}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}')
    else:
        args.epochs = 150
        if is_regression:
            model = LinReg(input_size=input_size)
            criterion = nn.MSELoss()
        else:
            model = LogReg(input_size=input_size, output_size=1)
            criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X).flatten()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).flatten()
                val_loss = criterion(val_outputs, y_val).item()
                
                if is_regression:
                    val_preds = val_outputs.cpu().numpy()
                    val_r2 = r2_score(y_val.cpu().numpy(), val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val.cpu().numpy(), val_preds))
                    writer.add_scalar('Metrics/val_r2', val_r2, epoch)
                    writer.add_scalar('Metrics/val_rmse', val_rmse, epoch)
                    print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}')
                else:
                    if len(torch.unique(y)) <= 2:  # Check if binary classification
                        accuracy = ((outputs > 0.5) == y).float().mean()
                        val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                        val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
                        writer.add_scalar('Metrics/val_auroc', val_auroc, epoch)
                        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f} Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}')
                    else:
                        # For multi-class, would need different metrics
                        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
    
    # Load the best model for testing
    if is_regression:
        best_model = LinReg(input_size=input_size)
    else:
        best_model = LogReg(input_size=input_size, output_size=1)
    
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    
    # Testing
    with torch.no_grad():
        test_outputs = best_model(X_test).cpu().numpy().flatten()
        
        if is_regression:
            test_r2 = r2_score(y_test.cpu().numpy(), test_outputs)
            test_rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy(), test_outputs))
            writer.add_scalar('Metrics/test_r2', test_r2)
            writer.add_scalar('Metrics/test_rmse', test_rmse)
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
        else:
            probs = test_outputs
            label_preds = (probs > 0.5).astype(int)
            auroc = roc_auc_score(y_test.cpu().numpy(), probs)
            acc = (label_preds == y_test.cpu().numpy()).mean()
            
            writer.add_scalar('Metrics/test_auroc', auroc)
            writer.add_scalar('Metrics/test_accuracy', acc)
            
            print(f"Test AUROC: {auroc:.4f}")
            print(f"Test Accuracy: {acc:.4f}")
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()