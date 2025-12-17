import os
import sys
import torch
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference # <--- NUEVO IMPORT
from tqdm import tqdm

# Importar nuestro cargador de datos
# Truco para que Python encuentre el módulo src desde aquí
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.data.lidc_dataset import get_dataloader
except ImportError:
    # Fallback por si se ejecuta como modulo -m
    from src.data.lidc_dataset import get_dataloader

# --- CONFIGURACIÓN PARA CPU ---
DATA_DIR = "/home/marco/projects/lidc-analysis/data/processed"
MODEL_DIR = "/home/marco/projects/lidc-analysis/experiments"
MAX_EPOCHS = 20           
VAL_INTERVAL = 1         
BATCH_SIZE = 2           
LR = 1e-3                
ROI_SIZE = (96, 96, 96)  # Tamaño del cubo para inferencia (mismo que entrenamiento)

def train():
    # 1. Preparar
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_determinism(seed=0)
    device = torch.device("cpu")
    print(f"--- INICIANDO ENTRENAMIENTO EN: {device} ---")

    # 2. Cargar Datos
    print("Cargando datasets...")
    # Reduce num_workers a 0 para evitar bloqueos en CPU al depurar
    train_loader, val_loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)

    # 3. Modelo
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # 4. Loss & Optimizador
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 5. Bucle
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        
        progress_bar = tqdm(train_loader, desc=f"Entrenando", leave=False)
        
        for batch_data in progress_bar:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss /= step
        print(f"  > Loss Promedio Train: {epoch_loss:.4f}")

        # 6. Validación (CORREGIDA CON SLIDING WINDOW)
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            print("  > Validando...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # --- CORRECCIÓN CRÍTICA ---
                    # En lugar de pasar la imagen entera (que tiene tamaños raros),
                    # pasamos una ventana deslizante de 96x96x96.
                    # sw_batch_size=4 significa que procesa 4 cubitos a la vez.
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=ROI_SIZE, 
                        sw_batch_size=4, 
                        predictor=model,
                        overlap=0.25 # Solapamiento para suavizar bordes
                    )
                    # --------------------------
                    
                    val_outputs = (val_outputs > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                print(f"  > Metric (Dice Score) Val: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_metric_model.pth"))
                    print("  💾 ¡Nuevo récord! Modelo guardado.")
    
    print(f"\n--- FIN DEL ENTRENAMIENTO ---")
    print(f"Mejor Dice Score: {best_metric:.4f} en Epoch {best_metric_epoch}")

if __name__ == "__main__":
    train()
