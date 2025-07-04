import argparse
from torchvision import transforms
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from src.model.factory import ModelHandlerFactory

def get_default_transforms():
    """Get the default ImageNet transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def extract_features(dataloader, model):
    all_features = []
    all_labels = []
    all_paths = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.avgpool.register_forward_hook(get_activation("avgpool"))

    model.to(device)
    with torch.no_grad(): 
        for batch in dataloader:
            if len(batch) == 2:
                imgs, labels = batch
            else:
                imgs, labels, path = batch 
            imgs = imgs.to(device) 
            _ = model(imgs)  
            features = activation['avgpool'].squeeze().cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels)
            all_paths.extend(path)

    all_features = np.vstack(all_features)
    return all_features, all_labels, all_paths

def main(args):
    BATCH_SIZE = 64 
    if args.output_path is None:
        OUTPUT_CSV = f"/home/grotehans/xai_locality/data/feature_vectors_{args.model_type}.csv"
    else:
        OUTPUT_CSV = args.output_path

    model_handler = ModelHandlerFactory.get_handler(args.model_type, args.model_path)
    dataset = model_handler.load_data(args.data_path)

    model = model_handler.model
    model.eval()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    features, labels, paths = extract_features(dataloader, model)
    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels
    feature_df['path'] = paths
    feature_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Feature extraction completed and saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction from ImageNet dataset")
    parser.add_argument("--model_type", type=str, default="binary_inception_v3", help="Type of the model to use")
    parser.add_argument("--model_path", type=str, default=None , help="Path to the model")
    parser.add_argument("--data_path", type=str,default = "/home/grotehans/xai_locality/data/cats_vs_dogs/test",  help="Path to the model data")
    parser.add_argument("--output_path", type=str, help="Path to the save feature vectors of data")
    args = parser.parse_args()
    main(args)
