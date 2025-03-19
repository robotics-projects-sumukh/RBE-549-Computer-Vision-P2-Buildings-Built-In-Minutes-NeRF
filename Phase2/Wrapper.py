#!/usr/bin/env bash

import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import json
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch.nn.functional as F 

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.random.manual_seed(0)
data_name = "lego"

def load_images(image_dir):
    images = []
    if (data_name == "spidey"):
        for filename in sorted(os.listdir(image_dir)):
            if filename.startswith("image_0") and filename.endswith(".jpg"):
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (400, 400)) / 255
                images.append(img)
    else:
        for i, filename in enumerate(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, f"r_{i}.png")
            img = cv2.imread(img_path) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 400)) / 255
            images.append(img)
            if i == 199:
                break
    return images



def loadDataset(dataset_path, mode, device):
    dataset_path_train = os.path.join(dataset_path, data_name, 'transforms_train.json')
    with open(dataset_path_train, 'r') as json_file:
        dataset_info = json.load(json_file)

    image_paths = []
    rotation_matrices = []
    transform_matrices = []

    fov_x = dataset_info['camera_angle_x']
    scene_frames = dataset_info['frames']
    
    for scene_frame in scene_frames:
        image_paths.append(scene_frame['file_path'])
        rotation_matrices.append(scene_frame['rotation'])
        transform_matrices.append(scene_frame['transform_matrix'])
    
    training_dir = os.path.join(dataset_path, data_name, 'train')
    training_images = load_images(training_dir)
    training_images_tensor = np.array(training_images, dtype=np.float32)
    training_images_tensor = torch.from_numpy(training_images_tensor).to(device)

    transform_matrices_tensor = np.array(transform_matrices, dtype=np.float32)
    transform_matrices_tensor = torch.from_numpy(transform_matrices_tensor).to(device)

    H, W = training_images_tensor[0].shape[0], training_images_tensor[0].shape[1]
    focal_length_length = 0.5 * W / np.tan(0.5 * fov_x)

    camera_info = (H, W, focal_length_length)

    test_dataset_path = os.path.join(dataset_path, data_name, 'transforms_test.json')
    with open(test_dataset_path, 'r') as file:
        test_data = json.load(file)

    test_frames = test_data['frames']
    test_image_paths = []
    test_rotation_matrices = []
    test_transform_matrices = []

    for frame in test_frames:
        test_image_paths.append(frame['file_path'])
        test_rotation_matrices.append(frame['rotation'])
        test_transform_matrices.append(frame['transform_matrix'])

    test_image_dir = os.path.join(dataset_path, data_name, 'test')
    test_images_tensor = load_images(test_image_dir)
    test_images_tensor = np.array(test_images_tensor, dtype=np.float32)
    test_images_tensor = torch.from_numpy(test_images_tensor)
    test_transforms_tensor = np.array(test_transform_matrices, dtype=np.float32)
    test_transforms_tensor = torch.from_numpy(test_transforms_tensor)

    return camera_info, training_images_tensor, transform_matrices_tensor, test_transforms_tensor, test_images_tensor

def PixelToRay(camera_info, pose):
    """
    Input:
        camera_info: image width, height, camera matrix
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        config_args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    H, W, focal_length = camera_info

    grid_x, grid_y = torch.meshgrid(torch.linspace(0, W-1, W).to(pose), torch.linspace(0, H-1, H).to(pose), indexing='ij')

    # Transpose for correct orientation
    grid_x = grid_x.T
    grid_y = grid_y.T

    # Compute normalized pixel coordinates
    normalized_x = (grid_x - W/2) / focal_length
    normalized_y = (grid_y - H/2) / focal_length
    
    # Form direction vectors
    ray_vectors = torch.stack((normalized_x, -normalized_y, -torch.ones_like(normalized_x)), dim=-1)
    ray_vectors = ray_vectors[..., np.newaxis, :]
    
    # Apply camera rotation
    cam_rotation = pose[:3, :3]
    cam_position = pose[:3, -1].view(1, 1, 3)
    
    # Transform ray directions to world space
    world_ray_directions = torch.sum(ray_vectors*cam_rotation, dim=-1)
    # Normalize directions
    world_ray_directions = world_ray_directions/torch.linalg.norm(world_ray_directions, axis=-1, keepdim=True)
    # Set ray origins to camera position
    world_ray_origins = cam_position.expand(world_ray_directions.shape[0], world_ray_directions.shape[1], -1)

    return world_ray_directions, world_ray_origins

def generateBatch(ray_origins, ray_directions, ground_truth_colors, config_args, training_mode=True):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        config_args: get batch size related information
    Outputs:
        A set of rays origins, ray_vectors and gt colors
    """
    if training_mode:
        # Randomly sample rays during training
        sample_indices = np.random.choice(ray_origins.shape[0], 4096, replace=False)
        batch_origins = ray_origins[sample_indices].to(device)
        batch_directions = ray_directions[sample_indices].to(device)
        batch_targets = ground_truth_colors[sample_indices].to(device)
    else:
        # Use all rays for evaluation
        batch_origins = ray_origins.to(device)
        batch_directions = ray_directions.to(device)
        batch_targets = ground_truth_colors.to(device)
    
    return batch_origins, batch_directions, batch_targets

def collectAllRays(scene_images, camera_poses, camera_info, config_args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        config_args: get batch size related information
    Outputs:
        A set of rays origins, ray_vectors and gt colors
    """
    collected_origins = []
    collected_directions = []
    collected_colors = []

    img_height, img_width, focal_length = camera_info
    for img_idx in range(scene_images.shape[0]):
        current_image = scene_images[img_idx]
        current_pose = camera_poses[img_idx]

        directions, origins = PixelToRay(camera_info, current_pose)
        collected_origins.append(origins.view(-1, 3))
        collected_directions.append(directions.view(-1, 3))
        collected_colors.append(current_image.view(-1, 3))

    # Concatenate all rays
    all_origins = torch.cat(collected_origins, dim=0)
    all_directions = torch.cat(collected_directions, dim=0)
    all_colors = torch.cat(collected_colors, dim=0)

    return all_origins, all_directions, all_colors


def render(model, ray_origins, ray_directions, config_args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    samples_per_ray = config_args.n_sample
    if data_name == 'spidey':
        near_bound = 0.05
        far_bound = 1.0
    else:
        near_bound = config_args.near_bound
        far_bound = config_args.far_bound

    # Create sample points along each ray
    distance_samples = torch.linspace(near_bound, far_bound, samples_per_ray)
    distance_samples = distance_samples.expand(ray_origins.shape[0], samples_per_ray).clone().to(device)
    
    # Add random noise to sample positions for better convergence
    jitter = torch.rand(*ray_origins.shape[:-1], samples_per_ray) * (far_bound - near_bound) / samples_per_ray
    distance_samples += jitter.to(device)

    # Calculate 3D positions of all sample points
    sample_positions = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * distance_samples.unsqueeze(-1)
    flattened_positions = sample_positions.view(-1, 3)

    # Expand directions for each sample point
    expanded_directions = ray_directions.expand(samples_per_ray, ray_directions.shape[0], 3).transpose(0, 1).reshape(-1, 3)
    
    predicted_colors, predicted_densities = model(flattened_positions, expanded_directions)
    # Reshape predictions to match sample positions
    
    predicted_colors = predicted_colors.view(*sample_positions.shape[:-1], 3)
    predicted_densities = predicted_densities.view(*sample_positions.shape[:-1])
    
    # Use Volume rendering to get the final image
    delta_distances = torch.zeros_like(distance_samples)
    # Fill in the calculated distances into the new tensor, leaving the last entry as zero
    delta_distances[..., :-1] =  distance_samples[..., 1:] - distance_samples[..., :-1]
    # Now, set the last entry to 1e10 to simulate an "infinite" distance for the last sample
    delta_distances[..., -1] = 1e10
    
    # Calculate alpha values and transmittance
    alpha_values = 1.0 - torch.exp(-predicted_densities * delta_distances)
    transmittance_adjust = 1.0 - alpha_values + 1e-10

    # Calculate transmittance (T_i in the paper)
    padded_transmittance = torch.cat([torch.ones(*transmittance_adjust.shape[:-1], 1).to(device), transmittance_adjust], dim=-1)
    cumulative_transmittance = torch.cumprod(padded_transmittance, dim=-1)

    # Remove first element for exclusive product
    final_transmittance = cumulative_transmittance[..., 1:]

    # Calculate final weights
    final_weights = alpha_values * final_transmittance

    # Calculate final color as weighted sum
    final_color = torch.sum(final_weights.unsqueeze(-1) * predicted_colors, dim=-2)
    
    # Calculate depth as weighted sum of distances
    final_depth = torch.sum(final_weights * distance_samples, dim=-1)

    # Calculate accumulated opacity
    final_opacity = torch.sum(final_weights, dim=-1)

    return final_color
    

def loss(groundtruth, prediction):
    return F.mse_loss(groundtruth, prediction)
    

def train(scene_images, camera_poses, camera_info, config_args):
    model = NeRFmodel().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_args.lrate)
    gradient_scaler = torch.amp.GradScaler()

    # Initialize TensorBoard writer
    log_writer = SummaryWriter(config_args.logs_path)

    error_history = []
    best_error = float('inf')
    
    all_origins, all_directions, all_targets = collectAllRays(scene_images, camera_poses, camera_info, config_args)

    for iteration in range(config_args.max_iters):
        model.train()

        with torch.amp.autocast(device_type='cuda'):
            # Sample batch of rays
            batch_origins, batch_directions, batch_targets = generateBatch(all_origins, all_directions, all_targets, config_args)
            # Forward pass
            predicted_colors = render(model, batch_origins, batch_directions, config_args)
            # Calculate loss and metrics
            iteration_loss = loss(batch_targets, predicted_colors)
            quality_metric = calculatePSNR(batch_targets, predicted_colors)

            print(f" Iteration: {iteration}, Loss: {iteration_loss.item()}, PSNR: {quality_metric.item()}")

            # Backpropagation with mixed precision
            optimizer.zero_grad()
            gradient_scaler.scale(iteration_loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            # Track metrics
            error_history.append(iteration_loss.item())
            log_writer.add_scalar('Loss/training', iteration_loss.item(), iteration)
            log_writer.add_scalar('Metrics/PSNR', quality_metric.item(), iteration)
        
        # Save checkpoint periodically
        if iteration % 100 == 0:
            if iteration_loss.item() < best_error:
                best_error = iteration_loss.item()
                print(f"Saving model at iteration: {iteration}, Loss: {iteration_loss.item()}")
                checkpoint_data = {
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': gradient_scaler.state_dict(),
                    'loss': iteration_loss.item()
                }
                torch.save(checkpoint_data, f"{config_args.checkpoint_path}/model_{iteration}.pt")
    
    log_writer.close()

def render_image(model, test_ray_origins, test_ray_directions, test_gt, H, W, config_args):
    # Move all inputs to device at once
    test_ray_origins = test_ray_origins.to(device)
    test_ray_directions = test_ray_directions.to(device)
    
    # Determine optimal batch size based on GPU memory
    num_rays_per_batch = config_args.test_batch_size if hasattr(config_args, 'test_batch_size') else 4000
    num_total_rays = H * W
    num_batches = (num_total_rays + num_rays_per_batch - 1) // num_rays_per_batch  # Ceiling division
    
    # Pre-allocate output tensor on GPU
    rgb_pred_test = torch.empty((num_total_rays, 3), device=device)
    
    # Process in batches
    for i in range(num_batches):
        start_idx = i * num_rays_per_batch
        end_idx = min((i + 1) * num_rays_per_batch, num_total_rays)
        
        # Use with torch.cuda.amp.autocast() for mixed precision
        with torch.cuda.amp.autocast(enabled=config_args.use_mixed_precision if hasattr(config_args, 'use_mixed_precision') else False):
            test_origins, test_directions, _ = generateBatch(
                test_ray_origins[start_idx:end_idx], 
                test_ray_directions[start_idx:end_idx], 
                test_gt[start_idx:end_idx] if test_gt is not None else None, 
                config_args, 
                training_mode=False
            )
            
            # Render the batch
            pred = render(model, test_origins, test_directions, config_args)
            rgb_pred_test[start_idx:end_idx] = pred
    
    # Reshape and return
    pred_image = rgb_pred_test.view(H, W, 3).cpu().detach().numpy()
    return pred_image

def test(images, poses, camera_info, config_args):
    H, W, focal_length = camera_info
    
    # Load model once at the beginning
    if config_args.load_checkpoint:
        model = NeRFmodel().to(device)

        if data_name == 'lego':
            checkpoint = torch.load("Checkpoints/model_lego.pt", map_location=device)
        elif data_name == 'ship':
            checkpoint = torch.load("Checkpoints/model_ship.pt", map_location=device)
        else:
            checkpoint = torch.load("Checkpoints/model_spidey.pt", map_location=device)        
        # Fix the state dict by removing the "module." prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        
        # Load the fixed state dict
        model.load_state_dict(new_state_dict)
        
        # Only create DataParallel wrapper after loading weights
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    model.eval()
    
    # Generate all rays at once
    test_ray_origins, test_ray_directions, test_gt = collectAllRays(images, poses, camera_info, config_args)
    
    num_images = test_gt.shape[0] // (H*W)
    num_rays_per_image = H*W
    
    # Pre-allocate arrays for results
    PSNRs = torch.zeros(num_images)
    SSIMs = torch.zeros(num_images)
    frames = []
    
    with torch.no_grad():  # Disable gradient computation
        for index in range(num_images):
            print(f"Testing on image: {index}/{num_images}")
            
            # Extract ray data for current image
            start_idx = num_rays_per_image * index
            end_idx = num_rays_per_image * (index + 1)
            
            # Render the image
            pred_image = render_image(
                model, 
                test_ray_origins[start_idx:end_idx], 
                test_ray_directions[start_idx:end_idx], 
                test_gt[start_idx:end_idx] if test_gt is not None else None,
                H, W, config_args
            )
            
            # Convert ground truth to numpy
            gt_image = test_gt[start_idx:end_idx].view(H, W, 3).cpu().detach().numpy()
            
            # Store frame for GIF
            img = (255 * pred_image).astype(np.uint8)
            frames.append((255 * pred_image).astype(np.uint8))

            # save image 
            imageio.imsave(f"{config_args.images_path}/{data_name}_{index}.png", img)
            
            # Calculate metrics
            PSNRs[index] = calculatePSNR(gt_image, pred_image)
            SSIMs[index] = calculateSSIM(gt_image, pred_image)
    
    # Report results
    print(f"Mean PSNR: {torch.mean(PSNRs):.4f}")
    print(f"Mean SSIM: {torch.mean(SSIMs):.4f}")
    print("Frames: ", len(frames))
    gif_filename = config_args.gif_filename if hasattr(config_args, 'gif_filename') else 'data.gif'
    imageio.mimsave(gif_filename, frames, fps=config_args.fps if hasattr(config_args, 'fps') else 30)
    print(f"GIF saved as {gif_filename}")
        
    return PSNRs.mean().item(), SSIMs.mean().item(), frames


def calculatePSNR(ground_truth, prediction):
    ground_truth = torch.tensor(ground_truth)
    prediction = torch.tensor(prediction)
    mse = torch.mean((ground_truth - prediction) ** 2)
    return 10 * torch.log10(1.0 / mse)

def calculateSSIM(ground_truth, prediction):
    ground_truth = torch.tensor(ground_truth).permute(2, 0, 1).unsqueeze(0)
    prediction = torch.tensor(prediction).permute(2, 0, 1).unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(prediction, ground_truth)

def main(config_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_info, images, poses, test_poses, test_images = loadDataset(config_args.dataset_path, config_args.mode, device)

    if config_args.mode == 'train':
        print("Training Started ...")
        train(images, poses, camera_info, config_args)
    elif config_args.mode == 'test':
        print("Testing Started ...")
        config_args.load_checkpoint = True
        test(test_images, test_poses, camera_info, config_args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='spidey', help='Name of the dataset')
    parser.add_argument('--dataset_path',default="./Data",help="dataset path")
    parser.add_argument('--mode',default='test',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=192,help="number of sample per ray")
    parser.add_argument('--max_iters',default=200000,help="number of max iterations for training")
    parser.add_argument('--near_bound',default=2,help="near bound")
    parser.add_argument('--far_bound',default=6,help="far bound")
    parser.add_argument('--logs_path',default="./logs",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Checkpoints",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image",help="folder to store images")
    parser.add_argument('--epochs', default=1000, help="number of epochs")
    parser.add_argument('--plot', default=False, help="whether to plot images or not")
    parser.add_argument('--image_size', default=400, help="image size")
    parser.add_argument('--test_batch_size', type=int, default=4000, help='Batch size for testing')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision for faster computation')
    parser.add_argument('--plot_interval', type=int, default=5, help='Plot every N images')
    parser.add_argument('--save_gif', action='store_true', help='Save output as GIF')
    parser.add_argument('--gif_filename', type=str, default='lego.gif', help='Filename for output GIF')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for GIF')
    return parser

if __name__ == "__main__":
    parser = configParser()
    config_args = parser.parse_args()
    if not os.path.exists(config_args.logs_path):
        os.makedirs(config_args.logs_path)
    if not os.path.exists(config_args.checkpoint_path):
        os.makedirs(config_args.checkpoint_path)
    if not os.path.exists(config_args.images_path):
        os.makedirs(config_args.images_path)
    data_name = config_args.data_name
    main(config_args)