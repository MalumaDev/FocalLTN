import glob
import shutil
import os
import tempfile
from pathlib import Path

import yaml
from tqdm import tqdm
from wakepy import keep

# Percorso della directory dei file di configurazione
configs_dir = "configs_paper_sum"

# Trova tutti i file YAML nella directory dei file di configurazione
config_files = list(glob.glob(os.path.join(configs_dir, "*.yml")))
config_files.sort(key=lambda x: (x.split("_")[-1], x), reverse=True)

data_ratio = 1
with tempfile.TemporaryDirectory() as destination_folder, keep.running() as k:
    destination_folder = Path(destination_folder)
    # Loop attraverso i file di configurazione
    for config_file in tqdm(config_files):
        destination_file = destination_folder / Path(config_file).name
        print(config_file)  # Stampa il nome del file
        # Copia il file di configurazione nel file di destinazione
        # shutil.copy(config_file, destination_file)

        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config["data_ratio"] = data_ratio

        with open(destination_file, "w") as f:
            yaml.dump(config, f)

        # Esegui il comando di train.py
        os.system(f"python train.py {destination_file}")
