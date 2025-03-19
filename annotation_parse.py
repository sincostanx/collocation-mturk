import json
import re
import pandas as pd
from utils import get_mask, overlay_mask, image_grid_text
from PIL import Image
import glob
from natsort import natsorted
from pathlib import Path
import os
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import argparse

def read_annotation(path):
    with open(path, 'r') as file:
        annot = file.read()

    annot = annot.replace('taskAnswers:"', '"taskAnswers":')  # Fix the key wrapping
    annot = annot.replace('\\"', '"')  # Unescape the quotes
    annot = annot.replace('"\\n', '"')  # Remove unnecessary line breaks within strings
    annot = annot.replace('\\n', '')  # Remove newlines from the entire string
    annot = annot.replace('"[', '[').replace(']"', ']')  # Convert the inner JSON array to a list
    annot = re.sub(r'}"]\s+}', '}] }', annot)
    annot = json.loads(annot)["taskAnswers"][0]

    return annot

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="dataset750_candidate_reformat.csv")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_images_hit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)

    return parser

if __name__ == "__main__":
    args = create_args().parse_args()

    NUM_IMAGES_HIT = args.num_images_hit
    root = args.dir # "./annotation_all/raw_tan"
    paths = natsorted(list(glob.glob(os.path.join(root, "*.json"))))

    dataset_df = pd.read_csv(args.dataset_path)
    image_paths = dataset_df["img_path"].to_list()

    # outdir = "./annotation_all"
    def process_annot(path):
        annot = read_annotation(path)
        page_id = int(str(Path(path).stem).split("_")[-1])
        for i in range(NUM_IMAGES_HIT):
            image_id = page_id * NUM_IMAGES_HIT + i

            mask_save_path = os.path.join(args.outdir, "mask", f"{image_id}.png")
            overlay_save_path = os.path.join(args.outdir, "overlay", f"{image_id}.png")
            if os.path.exists(mask_save_path) and os.path.exists(overlay_save_path):
                continue

            image = Image.open(image_paths[image_id])
            mask = get_mask(annot[f"annotatedResult{i+1}"]["labeledImage"]["pngImageData"])
            output = overlay_mask(image, mask, color=(0, 255, 0, 0))
            
            os.makedirs(Path(mask_save_path).parent, exist_ok=True)
            mask.save(mask_save_path)
            
            os.makedirs(Path(overlay_save_path).parent, exist_ok=True)
            output.save(overlay_save_path)

    process_func = partial(process_annot)
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, paths), total=len(paths)))