import numpy as np
from PIL import Image
import argparse
import os
from pathlib import Path
import glob
from natsort import natsorted
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from functools import partial

# MIN_ANNOTATOR = 2

def combine_mask(paths, min_annotator):
    if len(paths) < min_annotator: return None, None

    masks = [np.array(Image.open(path)) for path in paths]
    masks = [(mask > 0).astype(np.uint8) for mask in masks]
    add_result = np.stack(masks).sum(axis=0)
    avg_result = add_result >= (min_annotator / 2)
    
    return add_result, avg_result

def visualize(bg_image, result):
    fig, ax = plt.subplots(figsize=(10, 20))
    plt.imshow(bg_image, extent=[0, result.shape[1], result.shape[0], 0])
    im = ax.imshow(result, cmap='viridis', alpha=0.6)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, ax=ax, cax=cax)

    return fig

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="dataset750_candidate_reformat.csv")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--min_annotator", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)

    return parser

if __name__ == "__main__":
    args = create_args().parse_args()

    dataset_df = pd.read_csv(args.dataset_path)
    image_paths = dataset_df["img_path"].to_list()

    def combine_annot(image_id):
        add_mask_path = os.path.join(args.outdir, "mask_add", f"{image_id}.npy")
        avg_mask_path = os.path.join(args.outdir, "mask_average", f"{image_id}.npy")
        add_overlay_path = os.path.join(args.outdir, "overlay_add", f"{image_id}.png")
        avg_overlay_path = os.path.join(args.outdir, "overlay_average", f"{image_id}.png")
        
        check = lambda x: os.path.exists(x)
        if not (check(add_mask_path) and check(avg_mask_path) and check(add_mask_path) and check(avg_overlay_path)):
            paths = natsorted(list(glob.glob(os.path.join(args.dir, f"label_*/mask/{image_id}.png"))))
            add_result, avg_result = combine_mask(paths, args.min_annotator)

            if (add_result is not None) and (avg_result is not None):
                # save mask
                os.makedirs(Path(add_mask_path).parent, exist_ok=True)
                os.makedirs(Path(avg_mask_path).parent, exist_ok=True)
                np.save(add_mask_path, add_result)
                np.save(avg_mask_path, avg_result)

                # visualize overlayed result
                os.makedirs(Path(add_overlay_path).parent, exist_ok=True)
                os.makedirs(Path(avg_overlay_path).parent, exist_ok=True)

                bg_image = np.array(Image.open(image_paths[image_id]))
                fig = visualize(bg_image, add_result)
                fig.savefig(add_overlay_path, bbox_inches="tight")

                fig = visualize(bg_image, avg_result)
                fig.savefig(avg_overlay_path, bbox_inches="tight")

    image_ids = list(range(len(image_paths)))
    process_func = partial(combine_annot)
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, image_ids), total=len(image_ids)))

    # for image_id in range(len(image_paths)):
    #     add_mask_path = os.path.join(args.outdir, "mask_add", f"{image_id}.npy")
    #     avg_mask_path = os.path.join(args.outdir, "mask_average", f"{image_id}.npy")
    #     add_overlay_path = os.path.join(args.outdir, "overlay_add", f"{image_id}.png")
    #     avg_overlay_path = os.path.join(args.outdir, "overlay_average", f"{image_id}.png")
        
    #     check = lambda x: os.path.exists(x)
    #     if check(add_mask_path) and check(avg_mask_path) and check(add_mask_path) and check(avg_overlay_path):
    #         continue

    #     paths = natsorted(list(glob.glob(os.path.join(args.dir, f"label_*/mask/{image_id}.png"))))
    #     add_result, avg_result = combine_mask(paths)

    #     # save mask
    #     os.makedirs(Path(add_mask_path).parent, exist_ok=True)
    #     os.makedirs(Path(avg_mask_path).parent, exist_ok=True)
    #     np.save(add_mask_path, add_result)
    #     np.save(avg_mask_path, avg_result)

    #     # visualize overlayed result
    #     os.makedirs(Path(add_overlay_path).parent, exist_ok=True)
    #     os.makedirs(Path(avg_overlay_path).parent, exist_ok=True)

    #     bg_image = np.array(Image.open(image_paths[image_id]))
    #     fig = visualize(bg_image, add_result)
    #     fig.savefig(add_overlay_path, bbox_inches="tight")

    #     fig = visualize(bg_image, avg_result)
    #     fig.savefig(avg_overlay_path, bbox_inches="tight")

    #     break


        