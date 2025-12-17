import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# Truco para importar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.lidc_dataset import get_dataloader

# --- CONFIGURACIÓN ---
MODEL_PATH = "/home/marco/projects/lidc-analysis/experiments/best_metric_model.pth"
DATA_DIR = "/home/marco/projects/lidc-analysis/data/processed"
DEVICE = "cpu"

def visualize_results():
    print(f"--- CARGANDO MODELO DE: {MODEL_PATH} ---")
    
    # 1. Reconstruir la misma arquitectura
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)
    
    # 2. Cargar los "pesos" aprendidos (el cerebro entrenado)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Modelo cargado exitosamente.")
    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo del modelo. ¿Terminó el entrenamiento?")
        return

    model.eval()
    
    # 3. Obtener un paciente de validación (que la IA nunca estudió)
    _, val_loader = get_dataloader(DATA_DIR, batch_size=1)
    
    # Buscamos un paciente que tenga tumor (para que la comparación sea interesante)
    print("Buscando un paciente con tumor en el set de validación...")
    
    target_batch = None
    for batch in val_loader:
        if batch["label"].sum() > 0: # Si tiene pixeles blancos (tumor)
            target_batch = batch
            break
    
    if target_batch is None:
        print("No encontré tumores en los primeros ejemplos, usando el último cargado.")
        target_batch = batch

    # 4. Inferencia (Predicción)
    image, label = target_batch["image"].to(DEVICE), target_batch["label"].to(DEVICE)
    
    print("🧠 La IA está pensando (Inferencia)...")
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image, 
            roi_size=(96, 96, 96), 
            sw_batch_size=4, 
            predictor=model,
            overlap=0.25
        )
        # Convertir probabilidad a binario (0 o 1)
        output = (output > 0.5).float()

    # 5. Visualización (Corte central del tumor)
    # Buscamos el corte Z donde el tumor es más grande en la etiqueta real
    # Convertimos a numpy para manejarlo fácil
    lbl_np = label.cpu().numpy()[0, 0, :, :, :]
    if lbl_np.sum() > 0:
        # Encontrar el centro de masa del tumor real
        z_indices = np.where(lbl_np > 0)[2]
        slice_idx = int(np.mean(z_indices))
    else:
        slice_idx = lbl_np.shape[2] // 2 # Mitad del pulmón si no hay tumor
        
    print(f"Visualizando corte Z: {slice_idx}")

    img_slice = image.cpu().numpy()[0, 0, :, :, slice_idx]
    lbl_slice = label.cpu().numpy()[0, 0, :, :, slice_idx]
    pred_slice = output.cpu().numpy()[0, 0, :, :, slice_idx]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("TAC Original")
    plt.imshow(img_slice.T, cmap="gray", origin="lower")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Verdad (Radiólogo)")
    plt.imshow(img_slice.T, cmap="gray", origin="lower")
    plt.imshow(lbl_slice.T, cmap="jet", alpha=0.5, origin="lower") # Tumor en color
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicción (IA)")
    plt.imshow(img_slice.T, cmap="gray", origin="lower")
    plt.imshow(pred_slice.T, cmap="spring", alpha=0.5, origin="lower") # Predicción en otro color
    plt.axis("off")

    plt.savefig("resultado_inferencia.png")
    print("📸 ¡Foto tomada! Revisa el archivo 'resultado_inferencia.png'")

if __name__ == "__main__":
    visualize_results()
