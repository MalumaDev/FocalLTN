import glob
import shutil
import os
import tempfile

import yaml
from tqdm import tqdm


def main(folder):
    # Percorso della directory dei file di configurazione
    configs_dir = "configs_paper_gamma"

    # Percorso del file di destinazione
    destination_file = folder / "config.yml"

    # Trova tutti i file YAML nella directory dei file di configurazione
    config_file = list(glob.glob(os.path.join(configs_dir, "*.yml")))[0]
    seeds = [1300, 1301, 1302, 1303, 1304]
    gamma = [0, 1, 6]

    shutil.copy(config_file, destination_file)
    # Loop attraverso i file di configurazione
    with tqdm(total=len(seeds) * len(gamma)) as pbar:
        for seed in seeds:
            for g in gamma:
                # Carica il file di configurazione
                with open(destination_file, "w+") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    # Modifica il file di configurazione
                    config["random_seed"] = seed
                    config["gamma"] = g
                    yaml.dump(config, f)

                # Esegui il comando di train.py
                os.system(f"python train.py {destination_file}")
                os.system('cls' if os.name == 'nt' else 'clear')
                pbar.update(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(tmp)
