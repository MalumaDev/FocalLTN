import glob
import shutil
import os

import yaml
from tqdm import tqdm

# Percorso della directory dei file di configurazione
configs_dir = "configs_paper_focal"

# Percorso del file di destinazione
destination_file = "config.yml"

# Trova tutti i file YAML nella directory dei file di configurazione
config_files = list(glob.glob(os.path.join(configs_dir, "*.yml")))

config_files.sort(key=lambda x: (x.split("_")[-1], x))

# Loop attraverso i file di configurazione
for config_file in tqdm(config_files):
    print(config_file)  # Stampa il nome del file
    # Copia il file di configurazione nel file di destinazione
    # shutil.copy(config_file, destination_file)

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data_ratio"] = 0.5

    with open(destination_file, "w") as f:
        yaml.dump(config, f)

    # Esegui il comando di train.py
    os.system("python train.py")
