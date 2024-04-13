import glob
import shutil
import os

from tqdm import tqdm

# Percorso della directory dei file di configurazione
configs_dir = "configs_paper_focal"

# Percorso del file di destinazione
destination_file = "config.yml"

# Trova tutti i file YAML nella directory dei file di configurazione
config_files = glob.glob(os.path.join(configs_dir, "*.yml"))

# Loop attraverso i file di configurazione
for config_file in tqdm(config_files):
    print(config_file)  # Stampa il nome del file
    # Copia il file di configurazione nel file di destinazione
    shutil.copy(config_file, destination_file)
    # Esegui il comando di train.py
    os.system("python train.py")
