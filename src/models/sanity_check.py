import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from monai.networks.nets import UNet
from monai.losses import DiceLoss

# --- IMPORTACIONES DEL PROYECTO ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.lidc_dataset import get_dataloader

# --- CONFIGURACIÓN DE FUERZA BRUTA ---
DATA_DIR = "/home/marco/projects/lidc-analysis/data/processed"
DEVICE = "cpu"
EPOCHS = 600        # Aumentado para garantizar convergencia
LOG_FILE = "sanity_trace.csv"

def sanity_check():
    print(f"--- INICIANDO SANITY CHECK: PROTOCOLO FUERZA BRUTA (600 Epochs) ---")
    print(f"--- TRAZA DE DATOS: {LOG_FILE} ---")
    
    # 1. Obtener Datos
    train_loader, _ = get_dataloader(DATA_DIR, batch_size=1)
    train_iter = iter(train_loader)
    
    single_batch = None
    print("🔎 Buscando un paciente con un tumor GRANDE (>500 px)...")
    
    for i in range(200):
        try:
            batch = next(train_iter)
            if batch["label"].sum() > 500: 
                single_batch = batch
                print(f"✅ ¡Encontrado en intento {i+1}!")
                break
        except StopIteration:
            break
            
    if single_batch is None:
        print("❌ No se encontró tumor grande. Usando el último disponible.")
        if single_batch is None: return

    # --- SELECCIÓN DEL MEJOR CROP ---
    inputs_full = single_batch["image"].to(DEVICE)
    labels_full = single_batch["label"].to(DEVICE)
    
    # Elegimos el crop con más tumor
    best_idx = 1 if labels_full[1].sum() > labels_full[0].sum() else 0
    tumor_size = labels_full[best_idx].sum().item()
    
    inputs = inputs_full[best_idx].unsqueeze(0)
    labels = labels_full[best_idx].unsqueeze(0)
    
    print(f"🎯 Entrenando con Crop #{best_idx}. Tamaño Tumor: {tumor_size:.0f} voxeles")
    print(f"📊 Proporción Tumor/Fondo: {(tumor_size/(96*96*96))*100:.4f}%")

    # Validación de etiquetas
    if torch.unique(labels).max() > 1:
        labels = (labels > 0).float()

    # 2. Modelo
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)
    
    # Learning Rate agresivo
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_function = DiceLoss(sigmoid=True)

    # 3. Entrenamiento con Trazabilidad
    print(f"🚀 Iniciando bucle de {EPOCHS} épocas...")
    model.train()
    losses = []
    
    # Preparar archivo CSV para logs
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'Delta', 'Time_Sec'])
        
        start_time = time.time()
        prev_loss = 1.0
        
        for i in range(EPOCHS):
            epoch_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            # Calcular Delta (Mejora)
            delta = prev_loss - current_loss
            prev_loss = current_loss
            
            # Guardar en CSV
            writer.writerow([i+1, f"{current_loss:.6f}", f"{delta:.8f}", f"{time.time()-start_time:.2f}"])
            
            # Log en consola cada 20 épocas o si hay una mejora masiva
            if (i+1) % 50 == 0 or (i < 50 and (i+1) % 10 == 0):
                status = "🟢 Mejorando" if delta > 0 else "🔴 Estancado"
                if delta > 0.01: status = "🔥 SALTO GRANDE"
                
                print(f"Epoch {i+1}/{EPOCHS} | Loss: {current_loss:.4f} | Delta: {delta:.6f} | {status}")

    total_time = time.time() - start_time
    print(f"⏱️ Tiempo total: {total_time:.1f} segundos ({total_time/EPOCHS:.3f} s/epoch)")
    print(f"📉 Loss Final: {losses[-1]:.6f}")

    # 4. Visualización
    print("Generando imagen de diagnóstico...")
    model.eval()
    with torch.no_grad():
        final_output = model(inputs)
        final_pred = (final_output > 0.5).float()

    lbl_np = labels.cpu().numpy()[0, 0, :, :, :]
    z_indices = np.where(lbl_np > 0)[2]
    slice_idx = int(np.mean(z_indices))
    
    plt.figure(figsize=(15, 5))
    
    # Gráfica de Loss
    plt.subplot(1, 3, 1)
    plt.title(f"Curva de Aprendizaje (Min Loss: {min(losses):.4f})")
    plt.plot(losses)
    plt.grid(True, which='both')
    plt.xlabel("Epochs")
    plt.ylabel("Dice Loss")

    # Verdad
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Tumor)")
    img_show = inputs.cpu().numpy()[0, 0, :, :, slice_idx].T
    lbl_show = labels.cpu().numpy()[0, 0, :, :, slice_idx].T
    plt.imshow(img_show, cmap="gray", origin="lower")
    plt.imshow(lbl_show, cmap="jet", alpha=0.5, origin="lower")
    plt.axis("off")

    # Predicción
    plt.subplot(1, 3, 3)
    plt.title(f"Predicción IA (Epoch {EPOCHS})")
    pred_show = final_pred.cpu().numpy()[0, 0, :, :, slice_idx].T
    plt.imshow(img_show, cmap="gray", origin="lower")
    plt.imshow(pred_show, cmap="spring", alpha=0.5, origin="lower")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("sanity_check_turbo.png")
    print("✅ ¡LISTO! Revisa 'sanity_check_turbo.png' y 'sanity_trace.csv'")

if __name__ == "__main__":
    sanity_check()
