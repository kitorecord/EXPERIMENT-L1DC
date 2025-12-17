import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from tqdm import tqdm  # Barra de progreso

# --- CONFIGURACIÓN ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.lidc_dataset import get_dataloader

DATA_DIR = "/home/marco/projects/lidc-analysis/data/processed"
MODEL_PATH = "/home/marco/projects/lidc-analysis/experiments/best_metric_model.pth"
OUTPUT_DIR = "/home/marco/projects/lidc-analysis/results_batch"
DEVICE = "cpu"

def run_batch_inference():
    print(f"--- INICIANDO INFERENCIA MASIVA ---")
    
    # 1. Preparar Directorio de Salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Resultados se guardarán en: {OUTPUT_DIR}")

    # 2. Cargar Datos (Validación)
    # batch_size=1 para evaluar uno por uno
    _, val_loader = get_dataloader(DATA_DIR, batch_size=1)
    print(f"📊 Total pacientes a evaluar: {len(val_loader)}")

    # 3. Cargar Modelo
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Modelo cargado correctamente.")
    except FileNotFoundError:
        print("❌ ERROR: No existe el modelo best_metric_model.pth")
        return

    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Lista para el reporte Excel/CSV
    report_data = []

    # 4. Bucle Masivo
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Procesando"):
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # Inferencia
            outputs = sliding_window_inference(
                inputs, (96, 96, 96), 4, model, overlap=0.25
            )
            outputs = (outputs > 0.5).float()
            
            # Calcular Score individual
            dice_metric.reset()
            dice_metric(y_pred=outputs, y=labels)
            score = dice_metric.aggregate().item()
            
            # Datos del paciente
            tumor_size = labels.sum().item()
            pred_size = outputs.sum().item()
            has_tumor = tumor_size > 0
            
            # 5. Guardar Imagen (Snapshot del corte central)
            # Solo guardamos si hay tumor real O si la IA inventó algo grande
            if has_tumor or pred_size > 100:
                patient_id = f"patient_{i:03d}"
                
                # Buscar corte central
                if has_tumor:
                    z_idx = int(np.mean(np.where(labels.cpu().numpy()[0, 0] > 0)[2]))
                else:
                    z_idx = inputs.shape[4] // 2
                
                # Plot
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1)
                plt.title(f"TAC")
                plt.imshow(inputs.cpu().numpy()[0, 0, :, :, z_idx].T, cmap="gray", origin="lower")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title(f"Verdad")
                plt.imshow(inputs.cpu().numpy()[0, 0, :, :, z_idx].T, cmap="gray", origin="lower")
                plt.imshow(labels.cpu().numpy()[0, 0, :, :, z_idx].T, cmap="jet", alpha=0.5, origin="lower")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title(f"IA (Dice: {score:.4f})")
                plt.imshow(inputs.cpu().numpy()[0, 0, :, :, z_idx].T, cmap="gray", origin="lower")
                plt.imshow(outputs.cpu().numpy()[0, 0, :, :, z_idx].T, cmap="spring", alpha=0.5, origin="lower")
                plt.axis('off')
                
                plt.savefig(f"{OUTPUT_DIR}/{patient_id}_result.png")
                plt.close() # Importante para liberar memoria
            
            # Agregar al reporte
            report_data.append({
                "Patient_ID": i,
                "Has_Tumor": has_tumor,
                "Tumor_Size_Px": tumor_size,
                "Prediction_Size_Px": pred_size,
                "Dice_Score": score,
                "Status": "OK" if score > 0.5 else ("Partial" if score > 0 else "Fail")
            })

    # 6. Guardar Reporte Final
    df = pd.DataFrame(report_data)
    csv_path = f"{OUTPUT_DIR}/final_report.csv"
    df.to_csv(csv_path, index=False)
    
    print("-" * 30)
    print(f"🏁 PROCESO TERMINADO")
    print(f"📄 Reporte guardado en: {csv_path}")
    print(f"🖼️ Imágenes guardadas en: {OUTPUT_DIR}")
    print(f"📈 Score Promedio Global: {df['Dice_Score'].mean():.4f}")

if __name__ == "__main__":
    run_batch_inference()
