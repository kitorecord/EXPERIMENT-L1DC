import os
import json
import random
from pathlib import Path

# --- CONFIGURACIÓN ---
# Ajusta esta ruta si tu usuario no es 'marco'
DATA_DIR = "/home/marco/projects/lidc-analysis/data/processed"
OUTPUT_JSON = "dataset_split.json"
VAL_PERCENT = 0.2  # 20% de los datos para validación
SEED = 42          # Semilla para que el resultado sea siempre igual

def create_splits():
    images_dir = Path(DATA_DIR) / "images"
    
    # 1. Buscar archivos procesados (.nii.gz)
    # Obtenemos solo el ID (ej: LIDC-IDRI-0001) quitando la extensión
    patients = [f.name.replace(".nii.gz", "") for f in images_dir.glob("*.nii.gz")]
    
    # Ordenar para consistencia antes de mezclar
    patients.sort()
    
    if not patients:
        print(f"❌ ERROR: No se encontraron archivos en {images_dir}")
        print("¿Ya terminaste de ejecutar prepare_dataset.py?")
        return

    print(f"--- GENERANDO SPLITS ---")
    print(f"Total pacientes encontrados: {len(patients)}")

    # 2. Mezclar aleatoriamente
    random.seed(SEED)
    random.shuffle(patients)
    
    # 3. Calcular división
    split_idx = int(len(patients) * (1 - VAL_PERCENT))
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]
    
    # 4. Crear estructura JSON para MONAI
    # Usamos rutas relativas "./images/..."
    training_data = []
    validation_data = []
    
    for pid in train_patients:
        training_data.append({
            "image": f"./images/{pid}.nii.gz",
            "label": f"./masks/{pid}.nii.gz"
        })
        
    for pid in val_patients:
        validation_data.append({
            "image": f"./images/{pid}.nii.gz",
            "label": f"./masks/{pid}.nii.gz"
        })
        
    data = {
        "training": training_data,
        "validation": validation_data
    }
    
    # 5. Guardar archivo
    output_path = os.path.join(DATA_DIR, OUTPUT_JSON)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Archivo guardado en: {output_path}")
    print(f"   Entrenamiento: {len(train_patients)} pacientes")
    print(f"   Validación:    {len(val_patients)} pacientes")

if __name__ == "__main__":
    create_splits()
