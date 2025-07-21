# Importing the modules from the UNet folder
from DeepLabV3Plus.network.modeling import deeplabv3plus_resnet101
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms

import segmentation_models_pytorch as smp
from torchsummary import summary
from sklearn.metrics import f1_score

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
            gt_path = os.path.join(self.datadir, 'gts', image_name + '_gt.' + ext_name)

            self.imdb.append((img_path, gt_path))


    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):

        img_path, gt_path = self.imdb[idx]

        # Load images
        image = Image.open(img_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("L")  # Assuming GT is grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = np.array(gt_image)
        label = torch.LongTensor(label)  
        # label = torch.DoubleTensor(label)  

        return image, label, idx
    

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

def visualize_results(input_img, ground_truth, prediction, save_path="output_image.png"):
    """Displays the original input, ground truth, and prediction side by side and saves it"""
    plt.figure(figsize=(120, 40))

    # Input Image (keep it unchanged)
    plt.subplot(1, 3, 1)
    plt.imshow(input_img.astype(np.uint8))
    plt.title("Original Input", fontsize=90)
    plt.axis("off")

    # Ground Truth (apply color map)
    ground_truth_colored = color_map(ground_truth)
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_colored)
    plt.title("Ground Truth", fontsize=90)
    plt.axis("off")

    # Prediction (apply color map)
    prediction_colored = color_map(prediction)
    plt.subplot(1, 3, 3)
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


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    model.to(device)

    # Store patches for visualization
    patch_names = []
    input_patches = []    # Stores input patches
    gt_patches = []       # Stores ground truth patches
    pred_patches = []     # Stores predicted patches
    
    all_preds = []  # Store predictions for final F1 calculation
    all_targets = []  # Store ground truth for final F1 calculation


    with torch.no_grad():
        for images, labels, index in tqdm(loader):
            # for _, i in enumerate(index):
            #     og_img_path = os.path.join("BrickField/test/images", os.listdir("BrickField/test/images")[i])
            #     input_patches.append(np.array(Image.open(og_img_path)))  # Add input patches
            #     patch_names.append(og_img_path)  # Add input patches
            

            # plt.figure(figsize=(6, 6))
            # plt.imshow(np.array(Image.open(og_img_path)).astype(np.uint8))  # Assuming images are in (C, H, W) format
            # plt.title(f"Image Patch")
            # plt.axis("off")
            # plt.savefig(f"image_patch.png")  # Save image patch as a PNG
            # plt.close()

            images = images.to(device, dtype=torch.float32)
            # labels = labels.unsqueeze(1)
            labels = labels.to(device, dtype=torch.long)
            # labels = labels.to(device, dtype=torch.DoubleTensor)


            outputs = model(images)
            # outputs = outputs.argmax(dim=1)
            # outputs = outputs.unsqueeze(1)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()  

            # # Append patches
            # for _, target in enumerate(targets):
            #     gt_patches.append(target)  # Add ground truth patches
            # for _, pred in enumerate(preds):
            #     pred_patches.append(pred)  # Add predicted patches
                

            metrics.update(targets, preds)

            # Flatten each batch and append to lists
            all_preds.append(preds.flatten())  # Flatten to 1D per batch
            all_targets.append(targets.flatten())  # Flatten to 1D per batch

            # print(f"actual prediction shape = {preds.shape}")
            # print(f"total prediction list shape = {np.array(all_preds).shape}")

            # print(f"actual targets shape = {targets.shape}")
            # print(f"total targets list shape = {np.array(all_targets).shape}")


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

        # # Visualize the images side by side
        # visualize_results(full_input, full_gt, full_pred)

        score = metrics.get_results()

        # Concatenate all flattened arrays into a single 1D array
        all_preds = np.concatenate(all_preds)  # Shape: (total_pixels,)
        all_targets = np.concatenate(all_targets)  # Shape: (total_pixels,)
        
    return score, all_preds, all_targets

def train(args, model, train_loader, test_loader, device, restart=False):
    model.to(device)
    start_epoch = 0
    best_iou = 0.0  
    train_losses = []
    val_losses = []
    iou_scores = []

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

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
        project = "DeepLab V3+ Brickfield",
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

            # print(f"mask shape befpre unsqueeze: {masks.shape}")
            # print(f"mask data type before: {masks.dtype}")
            # masks = masks.unsqueeze(1)
            # print(f"masks shape after unsqueeze: {masks.shape}")
            # print(f"masks data type after: {masks.dtype}")

            optimizer.zero_grad()

            outputs = model(images)
            # print(f"pred shape before squeeze: {outputs.shape}")
            # print(f"pred data type before: {outputs.dtype}")
            # uni = torch.unique(outputs, dim=1)
            # print(uni)
            # outputs = outputs.argmax(dim=1)
            # outputs = outputs.unsqueeze(1)
            # outputs = outputs.to(dtype = torch.float64)
            # print(f"pred shape after squeeze: {outputs.shape}")
            # print(f"pred datatype after: {outputs.dtype}")


            loss = criterion(outputs, masks)
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # scheduler.step()

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
                # masks = masks.unsqueeze(1)

                outputs = model(images)
                # outputs = outputs.argmax(dim=1)
                # outputs = outputs.unsqueeze(1)
                # outputs = outputs.to(dtype = torch.float64, device = 'cuda')
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Compute IoU Score
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()  # Get predicted class
                targets = masks.cpu().numpy()
                metrics.update(targets, preds)

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

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
    test_dir = os.path.join(args.datadir, 'test')
    # test_dir = args.datadir

    # train_dataset = LULCDataset(train_dir, transform=transform, gt_transform = gt_transform)
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)

    # model = smp.Unet(
    # encoder_name="resnet50",        # Backbone pre-trained on ImageNet
    # encoder_weights="imagenet",     # Use ImageNet pre-trained weights
    # in_channels=3,                  # Input channels (3 for RGB images)
    # classes= args.num_classes,            # Number of output classesp
    # # dropout=0.5  # Add dropout with 50% rate
    # )
    
    #########################################################################
    # model = smp.Unet(
    #     encoder_name="resnet50",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=args.num_classes
    # )

    # # Dynamically add dropout to each decoder block
    # for block in model.decoder.blocks:
    #     block.dropout = nn.Dropout2d(p=0.5)
    #     original_forward = block.forward

    #     def new_forward(self, x, skip=None):
    #         x = original_forward(self, x, skip)
    #         return self.dropout(x)

    #     block.forward = new_forward.__get__(block, type(block))



    # Verify
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Dropout2d):
    #         print(f"Dropout found in {name} with rate {module.p}")

    # # Inspect encoder for dropout
    # encoder = model.encoder
    # for name, module in encoder.named_modules():
    #     if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
    #         print(f"Encoder Dropout in {name}: rate={module.p}")

    # # Inspect decoder for dropout
    # decoder = model.decoder
    # for name, module in decoder.named_modules():
    #     if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
    #         print(f"Decoder Dropout in {name}: rate={module.p}")

    # summary(model, input_size=(3, 512, 512))
    ####################################################################################


    model = deeplabv3plus_resnet101(pretrained_backbone=True, num_classes=args.num_classes)

    # Wrap the model with DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # print(model)
    # summary(model, input_size=(3,512,512))
    
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
    
    # # Train Set Metrics
    # train_metrics, all_preds, all_targets = validate(model=model, loader=train_loader, device=device, metrics=metrics)
    # logging.info('For train data: \n')
    # logging.info(metrics.to_str(train_metrics))

    # f1_classwise = f1_score(all_targets, all_preds, average=None) 
    # f1_micro = f1_score(all_targets, all_preds, average='micro')  
    # f1_macro = f1_score(all_targets, all_preds, average='macro')  
    # f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    # logging.info(f"\nF1 Scores - Non-Brickfield (0): {f1_classwise[0]:.4f}, Brickfield (1): {f1_classwise[1]:.4f}\n")
    # logging.info(f"\nF1 Score - Micro: {f1_micro:.4f}\n")
    # logging.info(f"\nF1 Score - Macro: {f1_macro:.4f}\n")
    # logging.info(f"\nF1 Score - Weighted: {f1_weighted:.4f}\n")

    # Test Set Metrics
    test_metrics, all_preds, all_targets = validate(model = model, loader=test_loader, device = device, metrics = metrics)
    # logging.info('For test data: \n')
    # logging.info(metrics.to_str(test_metrics))

    print('For test data: \n')
    print(metrics.to_str(test_metrics))


    # f1_classwise = f1_score(all_targets, all_preds, average=None) 
    # f1_micro = f1_score(all_targets, all_preds, average='micro')  
    # f1_macro = f1_score(all_targets, all_preds, average='macro')  
    # f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    # logging.info(f"\nF1 Scores - Non-Brickfield (0): {f1_classwise[0]:.4f}, Brickfield (1): {f1_classwise[1]:.4f}\n")
    # logging.info(f"\nF1 Score - Micro: {f1_micro:.4f}\n")
    # logging.info(f"\nF1 Score - Macro: {f1_macro:.4f}\n")
    # logging.info(f"\nF1 Score - Weighted: {f1_weighted:.4f}\n")

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

    parser = argparse.ArgumentParser(description="DeeplabV3+ training script")
    parser.add_argument("--datadir", type=str, default="dhaka_2019_data_512", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--num_classes", type = int, default=2, help="Number of classes")
    parser.add_argument("--checkpoint_path", type=str, default= 'checkpoint.pth', help='Saved model path')
    parser.add_argument("--best_model_path", type=str, default="checkpoint.pth",help="Path to save the best model checkpoint")#
    parser.add_argument("--losses_path", type=str, default="checkpoints/losses.pth",help="Path to all losses of model")#
    parser.add_argument("--eval_only", action='store_true', default=True, help = "Determining if only evaluation")
    parser.add_argument("--restart", default = False, action='store_true',  help="Determine if it should start from a checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="WandB Run ID for resuming training")

    # Parse command-line arguments
    args = parser.parse_args()
    # /teamspace/studios/this_studio/outputs/App.log
    # Call the main function with parsed arguments
    main(args)

