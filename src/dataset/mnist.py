import os
import numpy as np
from sklearn.model_selection import train_test_split

# Keep only PyTorch imports
try:
    import torch
    import torchvision
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def get_setting_name_mnist(n_samples=None, random_seed=42, binary_task=False, digits_subset=None, detect_eight=False):
    """Create a setting name for the MNIST dataset based on parameters."""
    setting_name = "mnist"
    if detect_eight:
        setting_name += "_binary_detect8"
    if n_samples is not None:
        setting_name += f"_n_samples{n_samples}"
    setting_name += f"_random_state{random_seed}"
    return setting_name

def create_mnist_tabular_data(data_folder,   test_size=0.6, val_size=0.3, random_seed=42, 
                             n_samples=None, binary_task=False, digits_subset=None,
                             detect_eight=False):
    """
    Create a tabular dataset from MNIST by flattening the images.
    
    Args:
        data_folder: Directory to save/load the data
        test_size: Proportion of original test data to use for testing
        val_size: Proportion of remaining train data to use for validation
        random_seed: Random seed for reproducibility
        n_samples: If specified, limit the dataset to this many samples
        binary_task: If True, convert to binary classification (requires digits_subset)
        digits_subset: List of digits to include, when None uses all digits
        detect_eight: If True, creates a binary task for detecting the digit 8
        
    Returns:
        tuple: (setting_name, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Override digits_subset if detect_eight is True
    if detect_eight:
        binary_task = True
        digits_subset = None  # We'll handle this differently for detect_eight
    
    setting_name = get_setting_name_mnist(n_samples, random_seed, binary_task, 
                                         digits_subset, detect_eight)
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    print("saving to: ", file_path)
    
    # If the processed data already exists, load it
    if os.path.exists(file_path):
        data = np.load(file_path)
        X_train = data['X_train']
        X_val = data['X_val'] 
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    else:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required to use the PyTorch MNIST dataset. "
                              "Please install them with 'pip install torch torchvision'.")
        # Use PyTorch's MNIST dataset
        train_images, train_labels, test_images, test_labels = load_mnist_from_torch(data_folder)
        
        # Handle detect_eight case: convert to binary classification (8 vs. rest)
        if detect_eight:
            # Convert to binary task: 1 if digit is 8, 0 otherwise
            train_labels = (train_labels == 8).astype(int)
            test_labels = (test_labels == 8).astype(int)
        # Filter by digits if needed (for other binary or subset tasks)
        elif digits_subset is not None:
            # Filter training data
            train_mask = np.isin(train_labels, digits_subset)
            train_images = train_images[train_mask]
            train_labels = train_labels[train_mask]
            
            # Filter test data
            test_mask = np.isin(test_labels, digits_subset)
            test_images = test_images[test_mask]
            test_labels = test_labels[test_mask]
            
            # Convert to binary task if requested
            if binary_task and len(digits_subset) == 2:
                train_labels = (train_labels == digits_subset[1]).astype(int)
                test_labels = (test_labels == digits_subset[1]).astype(int)
        
        # Limit the number of samples if specified
        if n_samples is not None and n_samples < len(train_images):
            # For binary tasks, stratify the sampling to maintain class balance
            stratify = train_labels if binary_task or detect_eight else None
            idx = np.random.RandomState(random_seed).choice(
                len(train_images), n_samples, replace=False, p=None if stratify is None 
                else np.ones(len(stratify))/len(stratify))
            train_images = train_images[idx]
            train_labels = train_labels[idx]
        # Split the original train data into train and temporary validation
        stratify = train_labels if binary_task or detect_eight else None
        X_train, X_temp_val, y_train, y_temp_val = train_test_split(
            train_images, train_labels, test_size=val_size, random_state=random_seed, 
            stratify=stratify)
        
        # Further split the temporary validation into final validation and additional test samples
        # Using the same proportion of splitting
        additional_test_fraction = test_size  # You can adjust this value as needed
        X_val, X_additional_test, y_val, y_additional_test = train_test_split(
            X_temp_val, y_temp_val, test_size=additional_test_fraction, random_state=random_seed,
            stratify=y_temp_val if stratify is not None else None)
        
        # Use the original test data augmented with additional samples from validation
        X_test = np.concatenate([test_images, X_additional_test])
        y_test = np.concatenate([test_labels, y_additional_test])
        
        # Save the processed data
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        print(file_path)
        np.savez(file_path, 
                X_train=X_train, X_val=X_val, X_test=X_test, 
                y_train=y_train, y_val=y_val, y_test=y_test)
    
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test

def create_detect_eight_task(data_folder, test_size=0.3, val_size=0.3, random_seed=42, n_samples=None):
    """
    Create a binary classification task specifically for detecting the digit 8.
    
    This is a convenience function that calls create_mnist_tabular_data with detect_eight=True.
    
    Args:
        data_folder: Directory to save/load the data
        test_size: Proportion of original test data to use for testing
        val_size: Proportion of remaining train data to use for validation
        random_seed: Random seed for reproducibility
        n_samples: If specified, limit the dataset to this many samples
        
    Returns:
        tuple: (setting_name, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    return create_mnist_tabular_data(
        data_folder=data_folder,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        n_samples=n_samples,
        detect_eight=True
    )

def load_mnist_from_torch(data_folder):
    """Load MNIST dataset using PyTorch's datasets with proper channel normalization."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision are required to use this function.")
    
    # Define transformations with normalization
    # MNIST is grayscale with mean=0.1307 and std=0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),  # Scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Channel normalization
    ])
    
    # Download and load training set
    train_dataset = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
    
    # Download and load test set
    test_dataset = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)
    
    # Convert to numpy arrays
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        # Convert each normalized image tensor to flattened numpy array
        train_images.append(img.numpy().reshape(-1))
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        # Convert each normalized image tensor to flattened numpy array
        test_images.append(img.numpy().reshape(-1))
        test_labels.append(label)
    
    # Convert lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    return train_images, train_labels, test_images, test_labels

def get_mnist_feature_names():
    """Get feature names for MNIST tabular data (pixel_0, pixel_1, etc.)"""
    return [f'pixel_{i}' for i in range(28*28)]

# Example usage:
if __name__ == "__main__":
    # Example: Create a binary classification task for detecting digit 8
    data_folder = "/home/grotehans/xai_locality/data"
    
    # Using the detect_eight option
    setting_name, X_train, X_val, X_test, y_train, y_val, y_test = create_mnist_tabular_data(
        data_folder, detect_eight=True)
    
    print(f"Dataset: {setting_name}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Class balance in training set: {np.mean(y_train)} (fraction of 8s)")
    
    # # Using the convenience function
    # print("\nUsing convenience function:")
    # setting_name, X_train, X_val, X_test, y_train, y_val, y_test = create_detect_eight_task(
    #     data_folder, n_samples=None)
    
    # print(f"Dataset: {setting_name}")
    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    # print(f"Class balance in training set: {np.mean(y_train)} (fraction of 8s)")
    
    # # Original binary classification example
    # print("\nBinary classification (0 vs 1):")
    # setting_name, X_train, X_val, X_test, y_train, y_val, y_test = create_mnist_tabular_data(
    #     data_folder, binary_task=True, digits_subset=[0, 1])
    
    # print(f"Dataset: {setting_name}")
    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")