#!/bin/bash
#SBATCH --job-name=WCagliari
#SBATCH --output=output_%A_%a.out 
#SBATCH --error=error_%A_%a.err    
#SBATCH --array=0-9               

# Percorsi dei file di input
DATASET="./unique_attendance_15/presenza_15_010824-140824.csv"
DATETIME_FILE="./unique_attendance_15/unique_dates_presenza_15_010824-140824.txt"

# Legge tutte le date uniche dal file in un array
mapfile -t unique_datetimes < "$DATETIME_FILE"

# Ottieni l'indice della data corrente per il job e anche la data successiva, se disponibile
DATETIME_1=${unique_datetimes[$SLURM_ARRAY_TASK_ID]}
DATETIME_2=${unique_datetimes[$((SLURM_ARRAY_TASK_ID + 1))]}

# Controlla che la data successiva sia definita (se si è vicini alla fine del file, non lo sarà)
if [[ -z "$DATETIME_2" ]]; then
    echo "Non esiste una data successiva per il job array ID $SLURM_ARRAY_TASK_ID. Terminazione anticipata."
    exit 1
fi

# Esegui lo script Python con entrambe le date come argomenti
python3 maxcut.py --datetime "$DATETIME_1" "$DATETIME_2" --dataset "$DATASET"
