#!/usr/bin/env python3

import os
import shutil


SAMPLES = 20
TIMEPOINTS = [0, 1, 2, 4, 8, 12]
ISOLATES = ["BPPQ.5", "EnFRAME01"]
SRC_DIR = "E. necator time progression"


def group_images():
    os.chdir(os.path.dirname(__file__))
    for isolate in ISOLATES:
        for i in range(SAMPLES):
            dest_dir = f"{isolate}-{i + 1}"
            os.makedirs(dest_dir, exist_ok=True)
            for timepoint in TIMEPOINTS:
                imgname = (
                    f"Tile00000{i + 1}.jpg" if i + 1 < 10 else f"Tile0000{i + 1}.jpg"
                )
                shutil.copy2(
                    f"{SRC_DIR}/{isolate} - T{timepoint}/{imgname}",
                    f"{dest_dir}/T{timepoint}.jpg",
                )


if __name__ == "__main__":
    group_images()
