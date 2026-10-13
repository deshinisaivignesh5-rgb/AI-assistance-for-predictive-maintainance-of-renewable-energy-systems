from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

WIND_URL = "https://raw.githubusercontent.com/n-sapkota/Fault-Detection-wind-turbine/main/wt.csv"
VISION_DATASET_REPO = "https://github.com/zae-bayern/elpv-dataset.git"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    wind_target = RAW_DIR / "wind_turbine_scada.csv"
    if not wind_target.exists():
        print("Downloading wind turbine SCADA dataset...")
        urlretrieve(WIND_URL, wind_target)
    else:
        print("Wind dataset already exists:", wind_target)

    vision_target = RAW_DIR / "elpv-dataset"
    if not vision_target.exists():
        if shutil.which("git") is None:
            raise RuntimeError("git is required to download the image-inspection dataset.")
        print("Cloning image-inspection dataset...")
        subprocess.run(
            ["git", "clone", "--depth", "1", VISION_DATASET_REPO, str(vision_target)],
            check=True,
        )
    else:
        print("Image-inspection dataset already exists:", vision_target)

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
