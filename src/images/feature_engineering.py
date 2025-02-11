import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path

from src.images.utils import load_image, calculate_normalization_stats


def resnet_representations(
    img_json: dict,
    *,
    model_name: str = "resnet50",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    normalize: bool = True,
    normalization_stats: dict = None,
    calculate_stats: bool = False,
) -> dict:
    """
    Extract image embeddings using a pre-trained ResNet model and save them to disk.
    Gets features from the penultimate layer (after average pooling, before classification),
    which are better suited for transfer learning and downstream tasks.
    Handles RGB, RGBA, and grayscale images.


    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        model_name: Name of the ResNet model to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        device: Device to run the model on ('cuda' or 'cpu')
        normalize: Whether to apply ImageNet normalization. Set to False if your images are significantly
                  different from natural images (e.g., medical, satellite, or microscopy images)
        normalization_stats: Optional dictionary with custom 'mean' and 'std' for normalization.
                           If None, ImageNet stats will be used when normalize=True
        calculate_stats: If True, calculate normalization statistics from the input images

    Returns:
        Updated img_json with embedding_paths field added containing paths to saved embeddings
    """
    # Validate model name
    valid_models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name. Choose from {valid_models}")

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Load pre-trained model
    model_fn = getattr(models, model_name)
    weights_name = (
        f"ResNet{model_name[6:]}_Weights"  # Convert 'resnet18' to 'ResNet18_Weights'
    )
    weights = getattr(models, weights_name).DEFAULT
    model = model_fn(weights=weights)

    # Remove the final classification layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Default ImageNet normalization values
    imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    # Build preprocessing pipeline
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    if calculate_stats:
        normalization_stats = calculate_normalization_stats(img_json["image_paths"])
        print(f"Calculated normalization stats: {normalization_stats}")

    if normalize:
        stats = normalization_stats or imagenet_stats
        transform_list.append(
            transforms.Normalize(mean=stats["mean"], std=stats["std"])
        )

    preprocess = transforms.Compose(transform_list)

    embedding_paths = {}

    with torch.no_grad():
        for img_path in img_json["image_paths"]:
            # Load and preprocess image
            image = load_image(img_path)
            image = Image.fromarray(image)

            # Handle different image modes
            if image.mode == "L":  # Grayscale
                # Convert to RGB by duplicating the channel
                image = image.convert("RGB")
            elif image.mode == "RGBA":
                # Remove alpha channel
                image = image.convert("RGB")
            elif image.mode not in ["RGB"]:
                raise ValueError(f"Unsupported image mode: {image.mode}")

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Extract features
            features = feature_extractor(input_batch)
            features = features.squeeze().cpu().numpy()

            # Create output filename based on input image path
            img_path_obj = Path(img_path)
            output_filename = f"{img_path_obj.stem}_{model_name}_representation.npz"
            output_path = artifacts_dir / output_filename

            # Save features as compressed numpy array
            np.savez_compressed(output_path, features=features)

            # Store path using image id as key
            img_id = img_json["path_to_id"][img_path]
            embedding_paths[img_id] = str(output_path)

    img_json["embedding_paths"] = embedding_paths
    img_json["embedding_metadata"] = {
        "model": model_name,
        "embedding_size": features.shape[0],
        "description": "Features extracted from penultimate layer (after average pooling, before classification)",
        "format": "numpy compressed array (.npz)",
        "normalization": {"applied": normalize, "stats": None},
    }

    # Set normalization stats based on what was used
    if normalize:
        img_json["embedding_metadata"]["normalization"]["stats"] = (
            normalization_stats if normalization_stats else imagenet_stats
        )

    return img_json


def vit_representations(
    img_json: dict,
    *,
    model_name: str = "vit_b_16",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    normalize: bool = True,
    normalization_stats: dict = None,
    calculate_stats: bool = False,
) -> dict:
    """
    Extract image embeddings using a pre-trained Vision Transformer model and save them to disk.
    Gets features from the final hidden state (before classification head).
    Handles RGB, RGBA, and grayscale images.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        model_name: Name of the ViT model to use ('vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14')
        device: Device to run the model on ('cuda' or 'cpu')
        normalize: Whether to apply ImageNet normalization. Set to False if your images are significantly
                  different from natural images (e.g., medical, satellite, or microscopy images)
        normalization_stats: Optional dictionary with custom 'mean' and 'std' for normalization.
                           If None, ImageNet stats will be used when normalize=True
        calculate_stats: If True, calculate normalization statistics from the input images

    Returns:
        Updated img_json with embedding_paths field added containing paths to saved embeddings
    """
    # Validate model name
    valid_models = ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name. Choose from {valid_models}")

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Load pre-trained model
    model_class = getattr(models, model_name, None)
    if model_class is None:
        raise ValueError(f"Model {model_name} not found in torchvision.models")

    weights_name = f"ViT_{model_name[4:].upper()}_Weights"
    weights = getattr(models, weights_name).DEFAULT
    model = model_class(weights=weights)

    # Remove the classification head
    feature_extractor = model
    feature_extractor.heads = nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Default ImageNet normalization values
    imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    ## Get input size based on model
    if "_14" in model_name:
        input_size = 224
    elif "_16" in model_name:
        input_size = 224
    else:  # '_32' models
        input_size = 224

    # Build preprocessing pipeline
    transform_list = [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]

    if calculate_stats:
        normalization_stats = calculate_normalization_stats(img_json["image_paths"])
        print(f"Calculated normalization stats: {normalization_stats}")

    if normalize:
        stats = normalization_stats or imagenet_stats
        transform_list.append(
            transforms.Normalize(mean=stats["mean"], std=stats["std"])
        )

    preprocess = transforms.Compose(transform_list)

    embedding_paths = {}

    with torch.no_grad():
        for img_path in img_json["image_paths"]:
            # Load and preprocess image
            image = load_image(img_path)
            image = Image.fromarray(image)

            # Handle different image modes
            if image.mode == "L":  # Grayscale
                # Convert to RGB by duplicating the channel
                image = image.convert("RGB")
            elif image.mode == "RGBA":
                # Remove alpha channel
                image = image.convert("RGB")
            elif image.mode not in ["RGB"]:
                raise ValueError(f"Unsupported image mode: {image.mode}")

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Extract features
            features = feature_extractor(input_batch)
            features = features.squeeze().cpu().numpy()

            # Create output filename based on input image path
            img_path_obj = Path(img_path)
            output_filename = f"{img_path_obj.stem}_{model_name}_representation.npz"
            output_path = artifacts_dir / output_filename

            # Save features as compressed numpy array
            np.savez_compressed(output_path, features=features)

            # Store path using image id as key
            img_id = img_json["path_to_id"][img_path]
            embedding_paths[img_id] = str(output_path)

    img_json["embedding_paths"] = embedding_paths
    img_json["embedding_metadata"] = {
        "model": model_name,
        "embedding_size": features.shape[0],
        "description": "Features extracted from final hidden state (before classification head)",
        "format": "numpy compressed array (.npz)",
        "normalization": {"applied": normalize, "stats": None},
    }

    # Set normalization stats based on what was used
    if normalize:
        img_json["embedding_metadata"]["normalization"]["stats"] = (
            normalization_stats if normalization_stats else imagenet_stats
        )

    return img_json
