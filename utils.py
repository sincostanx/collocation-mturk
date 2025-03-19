import base64
from PIL import Image, ImageFont, ImageDraw
import io
import numpy as np
from PIL import Image
import zipfile
import os
from pathlib import Path
import pandas as pd
import json

def return_pil_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return image_bytes.getvalue()

def get_mask(png_data):
    decoded_data = base64.b64decode(png_data)
    image = Image.open(io.BytesIO(decoded_data))
    image = np.array(image)
    image = Image.fromarray(255 * image)
    
    return image

def overlay_mask(image, mask, return_type="pil", color=(0, 255, 0, 0)):
    image = image.convert("RGBA")
    mask = mask.convert("L")

    # Create a transparent green overlay
    green_overlay = Image.new('RGBA', image.size, (*color[:3], 0))
    green_overlay.paste((*color[:3], 64), mask=mask)

    # Composite the green overlay with the original image
    result = Image.alpha_composite(image, green_overlay)

    if return_type == "pil":
        return result
    elif return_type == "byteio":
        return return_pil_image(result)
    else:
        raise NotImplementedError

def load_image(image_path: str, size=512, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    
    return Image.fromarray(image).resize((size, size))

def image_grid(imgs, rows, cols, size=(128, 128)):
    # assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    return grid

def image_grid_text(imgs, rows, cols, size=(128, 128), subtitles=None, font_size=20, font_path=None):
    # Calculate the size of each image
    w, h = imgs[0].size
    margin = 3  # Margin between text and image
    grid = Image.new('RGB', size=(cols * w, rows * (h + font_size + margin)), color=(255, 255, 255))
    grid_w, grid_h = grid.size

    # Load the font
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for i, img in enumerate(imgs):
        x = i % cols * w
        y = i // cols * (h + font_size + margin)
        grid.paste(img, box=(x, y + font_size + margin))

        # Draw the subtitle above the image if subtitles are provided
        if subtitles and i < len(subtitles):
            text_w, text_h = draw.textsize(subtitles[i], font=font)
            text_x = x + (w - text_w) // 2
            text_y = y + margin
            draw.text((text_x, text_y), subtitles[i], fill='black', font=font)

    return grid

def zip_files(file_list, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in file_list:
            print(os.path.join(*Path(file).parts[-2:]))
            zipf.write(file, arcname=os.path.join(*Path(file).parts[-2:]))
    print(f"Created ZIP file: {zip_name}")

def reformat_mturk_csv(df, N=10, start=None, end=None, prefix="https://mturk-collocation.s3.ap-southeast-2.amazonaws.com/small_dataset"):

    object_cols = [f"single_{i}" for i in range(2)] + [f"multiple_{i}" for i in range(2)]
    mturk_df = df[["img_path"] + object_cols]
    mturk_df["img_path"] = mturk_df["img_path"].apply(lambda x: os.path.join(prefix, os.path.join(*Path(x).with_suffix(".jpg").parts[-2:])))

    mturk_df = mturk_df.melt(id_vars=['img_path'], value_vars=object_cols, var_name='variable', value_name='object')
    mturk_df = mturk_df.drop(columns=['variable']).sort_values(by=['img_path', 'object']).reset_index(drop=True).dropna(subset="object")
    mturk_df = mturk_df.sample(frac=1, random_state=2024).reset_index(drop=True)

    mturk_df = mturk_df.head(len(mturk_df) // N * len(mturk_df))
    
    from copy import copy
    check_df = copy(mturk_df)

    if (start is not None) and (end is not None):
        if end > len(mturk_df): end = len(mturk_df)
        mturk_df = mturk_df[start:end]

    part_size = len(mturk_df) // N
    mturk_df = [mturk_df.iloc[i * part_size:(i + 1) * part_size].reset_index(drop=True) for i in range(N)]
    mturk_df = pd.concat(mturk_df, axis=1)

    import itertools
    mturk_df.columns = itertools.chain(*[[f"image_url{i+1}", f"labels{i+1}"] for i in range(N)])
    
    return mturk_df, check_df

def visualize_mturk(df, outdir, image_root="./data_preprocessed_jpg", num_images=10):
    image_cols = [f"Input.image_url{i+1}" for i in range(10)]
    object_cols = [f"Input.labels{i+1}" for i in range(10)]
    result_cols = [f"annotatedResult{i+1}" for i in range(10)]

    cols = image_cols + object_cols + result_cols + ["WorkerId"]
    num_cases = len(df)

    # change image_url to local path
    for col in image_cols:
        df[col] = df[col].apply(lambda x: os.path.join(image_root, *list(Path(x).parts[-2:])))

    # parse answer
    for col in result_cols:
        df[col] = df["Answer.taskAnswers"].apply(lambda x: get_mask(json.loads(x)[0][col]["labeledImage"]["pngImageData"]))

    for i in range(num_cases):
        data = df[cols].iloc[i].to_dict()
        worker_id = data["WorkerId"]
        results = []
        objects = []
        for j in range(num_images):
            image = Image.open(data[f"Input.image_url{j+1}"])
            mask = data[f"annotatedResult{j+1}"]
            result = overlay_mask(image, mask, color=(255, 0, 0, 0)).resize((256, 256))
            results.append(result)
            objects.append(data[f"Input.labels{j+1}"])

        save_path = os.path.join(outdir, f"label_{i:04d}_{worker_id}.png")
        os.makedirs(Path(save_path).parent, exist_ok=True)
        labels = image_grid(results, 4, 3, subtitles=objects, font_path="arial.ttf")
        labels.save(save_path)