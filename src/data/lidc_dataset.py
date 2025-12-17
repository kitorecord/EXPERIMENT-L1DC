import os
import json
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    Orientationd,
    Spacingd,
    ToTensord,
    SpatialPadd  # <--- NUEVO IMPORT
)
from monai.data import Dataset, DataLoader

def get_transforms(mode="train"):
    """
    Define las transformaciones (Augmentations) para MONAI.
    """
    # 1. Transformaciones base
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # --- CORRECCIÓN DE TAMAÑO ---
        # Si la imagen es menor a 96x96x96 (ej: 91x512x512), agregamos borde negro
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), method="symmetric"),
        # ----------------------------
    ]

    # 2. Transformaciones SOLO para entrenamiento
    if mode == "train":
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
        ])
    
    # 3. Convertir a Tensor
    transforms.append(ToTensord(keys=["image", "label"]))
    
    return Compose(transforms)

def get_dataloader(data_dir, split_json="dataset_split.json", batch_size=2):
    json_path = os.path.join(data_dir, split_json)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Falta {json_path}. Ejecuta src/data/create_splits.py")

    with open(json_path, "r") as f:
        data = json.load(f)
        
    train_files = [
        {"image": os.path.join(data_dir, x["image"]), "label": os.path.join(data_dir, x["label"])}
        for x in data["training"]
    ]
    val_files = [
        {"image": os.path.join(data_dir, x["image"]), "label": os.path.join(data_dir, x["label"])}
        for x in data["validation"]
    ]
    
    # Usamos Dataset normal (seguro para CPU)
    train_ds = Dataset(data=train_files, transform=get_transforms("train"))
    val_ds = Dataset(data=val_files, transform=get_transforms("val"))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
