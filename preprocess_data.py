import numpy as np
from PIL import Image
import glob
from pathlib import Path
import os
from tqdm.auto import tqdm
import argparse
from utils import load_image
from natsort import natsorted
from functools import partial
from multiprocessing import Pool

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default="*")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)

    return parser

if __name__ == "__main__":
    args = create_args().parse_args()

    def process_fn(args, path):
        # print(args)
        save_path = os.path.join(args.outdir, f"{Path(path).stem}.png")
        image = load_image(path, size=args.size)
        image.save(save_path)
        return None
    
    paths = natsorted(list(glob.glob(os.path.join(args.dir, args.glob_pattern))))
    os.makedirs(args.outdir, exist_ok=True)
    
    process_func = partial(process_fn, args)
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, paths), total=len(paths)))