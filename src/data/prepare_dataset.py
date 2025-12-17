print("--- [LOG] 1. Iniciando script... ---")

# 1. PARCHES DE COMPATIBILIDAD
try:
    print("--- [LOG] 2. Aplicando parches de compatibilidad Python 3.12/NumPy... ---")
    import configparser
    import numpy as np
    if not hasattr(configparser, 'SafeConfigParser'):
        configparser.SafeConfigParser = configparser.ConfigParser
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'bool'): np.bool = bool
    print("--- [LOG]    Parches aplicados correctamente. ---")
except Exception as e:
    print(f"!!! ERROR APLICANDO PARCHES: {e}")

# 2. IMPORTS
print("--- [LOG] 3. Importando librerías pesadas (Esto puede tardar unos segundos)... ---")
import os
import sys
import pandas as pd
import warnings
from tqdm import tqdm

print("--- [LOG]    Importando SimpleITK... ---")
import SimpleITK as sitk

print("--- [LOG]    Importando Pylidc (y conectando a DB)... ---")
import pylidc as pl
from pylidc.utils import consensus

warnings.filterwarnings("ignore")

# 3. CONFIGURACIÓN
OUTPUT_DIR = "/home/marco/projects/lidc-analysis/data/processed"
TARGET_SPACING = (1.0, 1.0, 1.0) 

# --- FUNCIONES ---
def resample_volume(volume, new_spacing, interpolator=sitk.sitkLinear):
    # (Misma lógica de siempre)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1024)
    resample.SetInterpolator(interpolator)
    return resample.Execute(volume)

def process_patient(scan):
    try:
        patient_id = scan.patient_id
        # A. Leer volumen
        vol_np = scan.to_volume(verbose=False)
        dicoms = scan.load_all_dicom_images(verbose=False)
        origin = [float(x) for x in dicoms[0].ImagePositionPatient]
        
        # B. Crear imagen base
        img_sitk = sitk.GetImageFromArray(vol_np)
        img_sitk.SetSpacing((scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing))
        img_sitk.SetOrigin(origin) 
        
        # C. Generar Máscara
        if len(scan.annotations) == 0:
            full_mask_np = np.zeros_like(vol_np, dtype=np.uint8)
        else:
            mask_vol, cbbox, masks = consensus(scan.annotations, clevel=0.5, pad=[(0,0), (0,0), (0,0)])
            full_mask_np = np.zeros_like(vol_np, dtype=np.uint8)
            full_mask_np[cbbox] = mask_vol.astype(np.uint8)
        
        mask_sitk = sitk.GetImageFromArray(full_mask_np)
        mask_sitk.CopyInformation(img_sitk)
        
        # D. Re-muestreo
        img_resampled = resample_volume(img_sitk, TARGET_SPACING, sitk.sitkLinear)
        mask_resampled = resample_volume(mask_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor)
        
        # E. Guardar
        sitk.WriteImage(img_resampled, f"{OUTPUT_DIR}/images/{patient_id}.nii.gz")
        sitk.WriteImage(mask_resampled, f"{OUTPUT_DIR}/masks/{patient_id}.nii.gz")
        
        return "OK"
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("--- [LOG] 4. Creando carpetas de salida... ---")
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)

    print("--- [LOG] 5. Consultando base de datos SQL (Puede congelarse aquí si la DB es lenta)... ---")
    # OPTIMIZACIÓN: No traemos todo el objeto, solo consultamos count primero para probar conexión
    try:
        count = pl.query(pl.Scan).count()
        print(f"--- [LOG]    Conexión exitosa. Hay {count} pacientes en la base de datos. ---")
        
        # Ahora sí traemos los objetos
        scans = pl.query(pl.Scan).all()
    except Exception as e:
        print(f"!!! ERROR FATAL EN BASE DE DATOS: {e}")
        sys.exit(1)
    
    # MODO PILOTO
    limit = None
    print(f"--- [LOG] 6. Iniciando bucle de procesamiento para los primeros {limit} pacientes... ---")
    
    scans_to_process = scans[:limit]
    
    log = []
    # Usamos tqdm pero forzamos que imprima updates seguido
    for i, scan in enumerate(scans_to_process):
        print(f"--- [LOG] Procesando {i+1}/{limit}: {scan.patient_id} ...")
        status = process_patient(scan)
        print(f"          > Resultado: {status}")
        
        log.append({"id": scan.patient_id, "status": status})

    print("--- [LOG] 7. Guardando CSV... ---")
    pd.DataFrame(log).to_csv("processing_results.csv", index=False)
    print("--- [LOG] FIN DEL SCRIPT ---")
