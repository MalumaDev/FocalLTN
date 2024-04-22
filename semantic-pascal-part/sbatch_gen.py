import glob
import shutil
import os
from pathlib import Path

import yaml
from tqdm import tqdm

# Percorso della directory dei file di configurazione
configs_dir = "FocalLTN/semantic-pascal-part/configs_paper_focal"

# Percorso del file di destinazione
destination_file = Path("FocalLTN/semantic-pascal-part/tmp")

destination_file.mkdir(parents=True,exist_ok=True)

# Trova tutti i file YAML nella directory dei file di configurazione
config_files = list(glob.glob(os.path.join(configs_dir, "*.yml")))

config_files.sort(key=lambda x: (x.split("_")[-1], x))

# Loop attraverso i file di configurazione
for n,config_file in enumerate(tqdm(config_files)):
    print(config_file)  # Stampa il nome del file
    # Copia il file di configurazione nel file di destinazione
    # shutil.copy(config_file, destination_file)

    out = destination_file / Path(config_file).name
    path = Path(f"tmp/{Path(config_file).name}.sbatch")
    path.parent.mkdir(parents=True,exist_ok=True)

    if path.exists():
        print(f"File {out} already exists. Exiting.")
        continue

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data_ratio"] = 0.5

    with open(out, "w") as f:
        yaml.dump(config, f)

    # Esegui il comando di train.py
    tmp = f"""#!/bin/bash
    #SBATCH --job-name=NeSy24{str(n)}
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=luca.piano@polito.it
    #SBATCH --time=4:00:00
    #SBATCH --nodes=1
    #SBATCH --mem=32000
    #SBATCH --ntasks-per-node=16
    #SBATCH --output=/home/lpiano/output_log/train_%j.log
    # module load intel/python/3
    source env/bin/activate
    

    export WANDB__SERVICE_WAIT=300

    cd FocalLTN/semantic-pascal-part

    python3 train.py {str(out)}"""

    path = f"tmp/{Path(config_file).name}.sbatch"
    with open(path, "w") as f:
        f.write(tmp)

    os.system(f"sbatch {str(path)}")


