#!/bin/bash

# --- CONFIGURACIÓN ---
PROJECT_DIR="/home/marco/projects/lidc-analysis"
BACKUP_DIR="/home/marco/backups"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FILENAME="lidc_code_backup_$TIMESTAMP.tar.gz"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}--- INICIANDO RESPALDO INTELIGENTE (Solo Código + Configs) ---${NC}"

# Crear carpeta si no existe
mkdir -p "$BACKUP_DIR"

# Entramos al directorio para que las rutas sean relativas
cd "$PROJECT_DIR"

# --- LA MAGIA: LISTA BLANCA ---
# Solo comprimimos explícitamente estas carpetas/archivos
echo "Comprimiendo..."

tar -czvf "$BACKUP_DIR/$FILENAME" \
    src \
    experiments \
    notebooks \
    *.sh \
    data/processed/dataset_split.json \
    2>/dev/null  # Ocultamos errores menores si no encuentra algún archivo

# Verificación
if [ $? -eq 0 ]; then
    SIZE=$(du -h "$BACKUP_DIR/$FILENAME" | cut -f1)
    echo -e "${GREEN}✅ ¡LISTO! Respaldo creado en segundos.${NC}"
    echo -e "Archivo: ${GREEN}$BACKUP_DIR/$FILENAME${NC}"
    echo -e "Tamaño:  $SIZE (Debería ser pequeño, unos MB)"
else
    echo -e "\033[0;31m❌ ERROR al comprimir.\033[0m"
fi
