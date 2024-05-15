import glob
import shutil
import os
import tempfile
from pathlib import Path

import yaml
from tqdm import tqdm
from wakepy import keep


def main(folder):
    # Percorso della directory dei file di configurazione
    configs_dir = "configs_paper_gamma"

    # Percorso del file di destinazione
    destination_file = Path(folder) / "config.yml"

    # Trova tutti i file YAML nella directory dei file di configurazione
    config_file = list(Path(configs_dir).glob("*.yml"))[0]
    seeds = [1300, 1301, 1302, 1303, 1304]
    seeds.reverse()
    types = ["focal_ltn", "focal_log_ltn"]
    gamma = [1, 6]

    shutil.copy(config_file, destination_file)
    with open(destination_file, "r") as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)

    # Loop attraverso i file di configurazione
    with tqdm(total=len(seeds) * len(gamma)) as pbar:
        for seed in seeds:
            for t in types:
                for g in gamma:
                    # Carica il file di configurazione
                    with open(destination_file, "w+") as f:
                        # Modifica il file di configurazione
                        general_config["random_seed"] = seed
                        general_config["gamma"] = g
                        general_config["ltn_config"] = t
                        yaml.dump(general_config, f)

                    # Esegui il comando di train.py
                    os.system(f"python train.py {destination_file}")
                    os.system('cls' if os.name == 'nt' else 'clear')
                    pbar.update(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, keep.running() as k:
        main(tmp)
