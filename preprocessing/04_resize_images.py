from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
import time

SOURCE = Path("raw_data/images")
DEST = Path("processed_images")
DEST.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.png"))

def process(path):
    output = DEST / path.name

    if output.exists():
        return

    with Image.open(path) as img:
        img = img.convert("L")
        img = ImageOps.pad(
            img,
            (224, 224),
            color=0,
            method=Image.Resampling.BILINEAR
        )
        img.save(output, format="PNG")

start = time.time()

with ThreadPoolExecutor(max_workers=8) as executor:
    for index, _ in enumerate(executor.map(process, files), 1):
        if index % 5000 == 0:
            print(
                f"Processed {index}/{len(files)} images "
                f"in {time.time() - start:.1f} seconds",
                flush=True
            )

print(f"Finished processing {len(files)} images!")
