import pandas as pd
from PIL import Image
from pathlib import Path
import os
import json
from utils import get_mask, overlay_mask, image_grid_text
import argparse
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_df", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="./input400_preprocessed_jpg")
    parser.add_argument("--num_images_hit", type=int, default=10)
    parser.add_argument("--save_image_grid", action="store_true")
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, Path(args.input_df).stem)

    df = pd.read_csv(args.input_df)
    image_cols = [f"Input.image_url{i+1}" for i in range(args.num_images_hit)]
    object_cols = [f"Input.labels{i+1}" for i in range(args.num_images_hit)]
    result_cols = [f"annotatedResult{i+1}" for i in range(args.num_images_hit)]
    cols = image_cols + object_cols + result_cols + ["WorkerId"]
    num_cases = len(df)

    # change S3 url to local path
    for col in image_cols:
        df[col] = df[col].apply(lambda x: os.path.join(args.image_root, *list(Path(x).parts[-1:])))

    # parse answer
    for col in result_cols:
        df[col] = df["Answer.taskAnswers"].apply(lambda x: get_mask(json.loads(x)[0][col]["labeledImage"]["pngImageData"]))

    all_results = {}
    for i in range(num_cases):
        data = df[cols].iloc[i].to_dict()
        worker_id = data["WorkerId"]
        results = []
        objects = []
        for j in range(args.num_images_hit):
            image_path = data[f"Input.image_url{j+1}"]
            image = Image.open(image_path)
            mask = data[f"annotatedResult{j+1}"]
            result = overlay_mask(image, mask, color=(0, 255, 0, 0))
            # result.save("temp.png")
            
            object_name = data[f"Input.labels{j+1}"]
            if (image_path, object_name) not in all_results:
                # print(worker_id)
                all_results[(image_path, object_name)] = [result]
            else:
                all_results[(image_path, object_name)].append(result)

            results.append(result.resize((256, 256)))
            objects.append(data[f"Input.labels{j+1}"])
            # print(result, len(results))

        if args.save_image_grid:
            save_path = os.path.join(args.outdir, f"grid_{i:04d}_{worker_id}.png")
            os.makedirs(Path(save_path).parent, exist_ok=True)
            labels = image_grid_text(results, 4, 3, subtitles=objects, font_path="arial.ttf")
            labels.save(save_path)
            # print(objects)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    for i, (k, v) in enumerate(all_results.items()):
        Image.open(k[0]).save(os.path.join(outdir, f"original_{i:04d}_{k[1]}_{0:03d}.png"))
        for j, image in enumerate(v):
            image.save(os.path.join(outdir, f"mask_{i:04d}_{k[1]}_{(j+1):03d}.png"))

    
        # image_grid_text