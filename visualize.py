import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import build_model
from dataloader.dataloader import getDataloader
import cv2 # Using OpenCV to read and resize the image file

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualization script for U-Bench")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use (e.g., U_Net)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory of the dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., busi)')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the visualization images')
    parser.add_argument('--gpu', type=str, default="0", help='GPU to use')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for the model')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--input_channel', type=int, default=3, help='Number of input channels')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for visualization (usually 1)')
    parser.add_argument('--do_deeps', type=bool, default=False, help='Use deep supervision')

    # Add dummy arguments that are present in the original main.py parser to avoid errors
    parser.add_argument('--train_file_dir', type=str, default="train.txt", help='train_file_dir')
    parser.add_argument('--val_file_dir', type=str, default="val.txt", help='val_file_dir')

    return parser.parse_args()

def load_model(args, device):
    """Loads a model and its checkpoint."""
    model = build_model(config=args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    return model

def visualize(args):
    """Generates and saves visualizations."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model and dataloader
    model = load_model(args, device)
    _, valloader = getDataloader(args)

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Saving visualizations to {args.save_path}")

    with torch.no_grad():
        for i, sampled_batch in enumerate(tqdm(valloader, desc="Generating visualizations")):
            image_tensor, label = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            case = sampled_batch.get('case', [f'image_{i}'])
            if isinstance(case, list):
                case = case[0]

            # Get model output and process it
            outputs = model(image_tensor)
            if args.do_deeps:
                outputs = outputs[-1]
            preds = (torch.sigmoid(outputs) > 0.5).float() if args.num_classes == 1 else torch.argmax(outputs, dim=1).unsqueeze(1)

            # Prepare ground truth and prediction for plotting
            label_np = label.cpu().numpy().squeeze()
            preds_np = preds.cpu().numpy().squeeze()

            # --- START OF CORRECTION ---
            # Re-load the original image from file to get the untransformed version
            original_image_path = os.path.join(args.base_dir, 'images', f"{case}.png")
            
            if not os.path.exists(original_image_path):
                print(f"Warning: Could not find original image at {original_image_path}. Skipping.")
                continue
            
            original_image = cv2.imread(original_image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Resize the original image to match the model's output dimensions
            target_shape = (label_np.shape[1], label_np.shape[0])  # Get shape (width, height) from label
            original_image_resized = cv2.resize(original_image, target_shape, interpolation=cv2.INTER_AREA)
            # --- END OF CORRECTION ---
            
            # Create plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.01)

            # Original Image (resized to match output)
            axes[0].imshow(original_image_resized)
            axes[0].set_title('Original Image', fontsize=16)
            axes[0].axis('off')

            # Ground Truth
            axes[1].imshow(label_np, cmap='gray')
            axes[1].set_title('Ground Truth', fontsize=16)
            axes[1].axis('off')

            # Model Prediction
            axes[2].imshow(preds_np, cmap='gray')
            axes[2].set_title('Model Prediction', fontsize=16)
            axes[2].axis('off')
            
            # Save as PDF with minimal padding
            save_filename = os.path.join(args.save_path, f"{os.path.basename(case).split('.')[0]}_visualization.pdf")
            plt.savefig(save_filename, format='pdf', bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

    print("Visualization complete.")

if __name__ == "__main__":
    args = parse_arguments()
    visualize(args)