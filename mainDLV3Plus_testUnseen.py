# Importing the modules from the UNet folder
from DeepLabV3Plus.network.modeling import deeplabv3plus_resnet101
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms

import segmentation_models_pytorch as smp
from torchsummary import summary

from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import logging
import argparse
import wandb

from DeepLabV3Plus.metrics import StreamSegMetrics

import matplotlib.pyplot as plt

class LULCDataset(Dataset):
    def __init__(self, datadir, transform=None, gt_transform = None):
        self.datadir = datadir
        self.transform = transform
        self.gt_transform = gt_transform

        self.imdb = []
        for img in os.listdir(self.datadir+'/images'):
            image_name = img.split('.')[0]
            ext_name = img.split('.')[-1]
            img_path = os.path.join(self.datadir, 'images', img)
            # gt_path = os.path.join(self.datadir, 'gts', image_name + '_gt.' + ext_name)

            # self.imdb.append((img_path, gt_path))
            self.imdb.append(img_path)


    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):

        # img_path, gt_path = self.imdb[idx]
        img_path = self.imdb[idx]

        # Load images
        image = Image.open(img_path).convert("RGB")
        # gt_image = Image.open(gt_path).convert("L")  # Assuming GT is grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # label = np.array(gt_image)
        # label = torch.LongTensor(label)  

        # return image, label, idx
        return image, idx

def stitch_patches(patches, num_patches, patch_size, is_image=False):
    """Stitches patches into a grid."""
    # Calculate grid size based on number of patches
    grid_size = int(np.floor(num_patches**0.5))  # Rows and columns (square grid)
    
    # Initialize the full image based on the grid size and patch size
    if is_image:
        full_image = np.zeros((patch_size * grid_size, patch_size * grid_size, 3), dtype=np.uint8)  # RGB
    else:
        full_image = np.zeros((patch_size * grid_size, patch_size * grid_size), dtype=np.uint8)  # Single-channel

    # Place patches in the full image grid
    index = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if index < len(patches):
                x_start = i * patch_size
                y_start = j * patch_size
                if is_image:
                    full_image[x_start:x_start + patch_size, y_start:y_start + patch_size, :] = patches[index]
                else:
                    full_image[x_start:x_start + patch_size, y_start:y_start + patch_size] = patches[index]
            index += 1

    return full_image

# def visualize_results(input_img, ground_truth, prediction, save_path="output_image.png"):
def visualize_results(input_img,  prediction, save_path="output_image.png"):
    """Displays the original input, ground truth, and prediction side by side and saves it"""
    plt.figure(figsize=(120, 60))

    # Input Image (keep it unchanged)
    plt.subplot(1, 2, 1)
    plt.imshow(input_img.astype(np.uint8))
    plt.title("Original Input", fontsize=90)
    plt.axis("off")

    # # Ground Truth (apply color map)
    # ground_truth_colored = color_map(ground_truth)
    # plt.subplot(1, 3, 2)
    # plt.imshow(ground_truth_colored)
    # plt.title("Ground Truth", fontsize=90)
    # plt.axis("off")

    # Prediction (apply color map)
    prediction_colored = color_map(prediction)
    plt.subplot(1, 2, 2)
    plt.imshow(prediction_colored)
    plt.title("Model Prediction", fontsize=90)
    plt.axis("off")

    plt.subplots_adjust(wspace=0.1)  # Reduce horizontal space between subplots

    # Save the figure to a file
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # Reduce whitespace around the image
    plt.close()

    print(f"Visualization saved to {save_path}")

def color_map(labels_or_preds):
    """Color map the labels or predictions where 0 is black and 1 is RGB(202, 99, 64)."""
    colored_image = np.zeros((labels_or_preds.shape[0], labels_or_preds.shape[1], 3), dtype=np.uint8)
    
    # Color the 1's with RGB(202, 99, 64) and the 0's remain black
    colored_image[labels_or_preds == 1] = [202, 99, 64]
    
    return colored_image

def validate(model, loader, device, metrics, path):
    """Do validation and return specified samples"""
    metrics.reset()
    model.to(device)

    # Store patches for visualization
    # patch_names = []
    # input_patches = []    # Stores input patches
    # gt_patches = []       # Stores ground truth patches
    # pred_patches = []     # Stores predicted patches

    os.makedirs(path + "/preds", exist_ok=True)

    with torch.no_grad():
        # for images, labels, index in tqdm(loader):
        for images, index in tqdm(loader):
            # for _, i in enumerate(index):
            #     og_img_path = os.path.join("test64/images", os.listdir("test64/images")[i])
            #     input_patches.append(np.array(Image.open(og_img_path)))  # Add input patches
            #     patch_names.append(og_img_path)  # Add input patches
            

            # plt.figure(figsize=(6, 6))
            # plt.imshow(np.array(Image.open(og_img_path)).astype(np.uint8))  # Assuming images are in (C, H, W) format
            # plt.title(f"Image Patch")
            # plt.axis("off")
            # plt.savefig(f"image_patch.png")  # Save image patch as a PNG
            # plt.close()

            images = images.to(device, dtype=torch.float32)
            # labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            # targets = labels.cpu().numpy()

            # # Append patches
            # for _, target in enumerate(targets):
            #     gt_patches.append(target)  # Add ground truth patches
            # for _, pred in enumerate(preds):
            #     pred_patches.append(pred)  # Add predicted patches

            # Save each prediction in the batch
            for i in range(len(preds)):
                pred = preds[i]  # Shape: [H, W]
                indice = index[i]  # Identifier for the image (e.g., file path or name)
                
                # Extract the base filename and ensure PNG extension
                og_img_path = os.path.join(path + "/images", os.listdir(path + "/images")[indice])
                base_name = os.path.basename(og_img_path)
                name, _ = os.path.splitext(base_name)
                save_name = name + ".png"
                save_path = os.path.join(path + "/preds", save_name)
                
                # Save the prediction as a PNG image
                Image.fromarray(color_map(pred.astype(np.uint8))).save(save_path)                

            # metrics.update(targets, preds)


        # sorted_indices = np.argsort(patch_names)

        # input_patches = [input_patches[i] for i in sorted_indices]
        # gt_patches = [gt_patches[i] for i in sorted_indices]
        # pred_patches = [pred_patches[i] for i in sorted_indices]

        # # Convert patches into full images
        # num_patches = len(input_patches)
        # patch_size = 1500  # Assumes all patches are the same size
        
        # full_input = stitch_patches(input_patches, num_patches, patch_size, is_image=True)
        # full_gt = stitch_patches(gt_patches, num_patches, patch_size)
        # full_pred = stitch_patches(pred_patches, num_patches, patch_size)

        # Visualize the images side by side
        # visualize_results(full_input, full_gt, full_pred)
        # visualize_results(full_input, full_pred)

        # score = metrics.get_results()
    # return score
    return None

def train(args, model, train_loader, test_loader, device, restart=False):
    model.to(device)
    start_epoch = 0

    best_iou = 0.0  

    train_losses = []
    val_losses = []
    iou_scores = []

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if restart:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Retrieve and set run_id **before initializing wandb**
        args.run_id = checkpoint.get("run_id", None)
        print(f"Resuming WandB run ID: {args.run_id}")  # Debugging print
    
    else:
        # If not restarting, remove old loss and best model files
        if os.path.exists(args.losses_path):
            os.remove(args.losses_path)
        if os.path.exists(args.best_model_path):
            os.remove(args.best_model_path)

    run = wandb.init(
        entity="samirashid5800-independent-university-bangladesh",
        project = "Unet LULC Brickfield",
        config={
            "batch_size" : args.batch_size,
            "epoch" : args.epochs,
            "learning_rate" : args.lr,
        },
        resume="allow" if args.run_id else None,  # Ensures resuming only when a run_id exists,
        id = args.run_id  # Resume by run_id
    )
    print(f"The run id is {run.id}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks, index in tqdm(train_loader, desc=f"Epoch: {epoch+1} / {args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Calculate average train loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        metrics = StreamSegMetrics(args.num_classes)  # IoU calculator
        with torch.no_grad():
            for images, masks, index in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Compute IoU Score
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()  # Get predicted class
                targets = masks.cpu().numpy()
                metrics.update(targets, preds)

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Compute IoU metrics
        iou_results = metrics.get_results()
        avg_iou = iou_results["Mean IoU"]  # Extract mean IoU
        print(f"mean_iou for epoch {epoch} is {avg_iou}")
        iou_scores.append(avg_iou)

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}")
        run.log({
                "epoch" : epoch+1,
                "Train_Loss": avg_train_loss,
                "Val_loss": avg_val_loss,
                "Mean_IoU": avg_iou
                })

        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'run_id': run.id  # Store the run ID
        }
        torch.save(checkpoint, args.checkpoint_path)

        # Save the best model based on IoU
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'run_id': run.id,  # Store the run ID
                'best_iou': best_iou,
                'best_train_loss': avg_train_loss,
                'best_val_loss': avg_val_loss
            }
            torch.save(best_checkpoint, args.best_model_path)

    # Save train/val losses & IoU scores
    loss_data = {
    'train_losses': train_losses if train_losses else ["No data"],
    'val_losses': val_losses if val_losses else ["No data"],
    'iou_scores': iou_scores if iou_scores else ["No data"]
    }
    torch.save(loss_data, args.losses_path) 


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225]) 
    ])
    gt_transform = transforms.Compose([    
        transforms.ToTensor()
    ])


    # train_dir = os.path.join(args.datadir, 'train')
    # test_dir = os.path.join(args.datadir, 'test')
    test_dir = args.datadir

    # train_dataset = LULCDataset(train_dir, transform=transform, gt_transform = gt_transform)
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)
    # model = smp.Unet(
    #     encoder_name="resnet50",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=args.num_classes
    # )
    model = deeplabv3plus_resnet101(pretrained_backbone=True, num_classes=args.num_classes)

    # Wrap the model with DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    if not args.eval_only:
        # train(args, model, train_loader, test_loader, device, restart = args.restart)
        pass

    if args.eval_only:
        checkpoint = torch.load(args.best_model_path, weights_only=False)
        # model.load_state_dict(checkpoint['model_state_dict'])

        # Adjust for multi-GPU loading if needed
        if torch.cuda.device_count() > 1:
            new_state_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                new_key = "module." + key if not key.startswith("module.") else key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    metrics = StreamSegMetrics(args.num_classes)
    model.eval()
    
    # train_metrics = validate(model=model, loader=train_loader, device=device, metrics=metrics)
    # logging.info('For train data: \n')
    # logging.info(metrics.to_str(train_metrics))

    test_metrics = validate(model = model, loader=test_loader, device = device, metrics = metrics, path = args.datadir)
    # logging.info('For test data: \n')
    # logging.info(metrics.to_str(test_metrics))
    print("Validation ended")

if __name__ == '__main__':

    # Set up logging with a file handler
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('outputs/App.log')  # Log to file "app.log"
        ]
    )

    logging.getLogger('PIL').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="DeeplabV3+ Training Script")
    parser.add_argument("--datadir", type=str, default="dhaka_2023/", help="Path to the dataset directory") #for bagerhat
    parser.add_argument("--batch_size", type=int, default=150, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--num_classes", type = int, default=2, help="Number of classes")
    parser.add_argument("--checkpoint_path", type=str, default= 'checkpoints/checkpoint.pth', help='Saved model path')
    parser.add_argument("--best_model_path", type=str, default="checkpoints/checkpoint.pth",help="Path to save the best model checkpoint")#
    parser.add_argument("--losses_path", type=str, default="checkpoints/losses.pth",help="Path to all losses of model")#
    parser.add_argument("--eval_only", action='store_true', default=True, help = "Determining if only evaluation")
    parser.add_argument("--restart", default = False, action='store_true',  help="Determine if it should start from a checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="WandB Run ID for resuming training")

    # Parse command-line arguments
    args = parser.parse_args()
    # /teamspace/studios/this_studio/outputs/App.log
    # Call the main function with parsed arguments
    main(args)

