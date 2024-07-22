import base64
from PIL import Image
import io
import numpy as np

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

def overlay_mask(image, mask, return_type="pil"):
    image = image.convert("RGBA")
    mask = mask.convert("L")

    # Create a transparent green overlay
    green_overlay = Image.new('RGBA', image.size, (0, 255, 0, 0))
    green_overlay.paste((0, 255, 0, 64), mask=mask)

    # Composite the green overlay with the original image
    result = Image.alpha_composite(image, green_overlay)

    if return_type == "pil":
        return result
    elif return_type == "byteio":
        return return_pil_image(result)
    else:
        raise NotImplementedError

def load_image(image_path: str, size=512, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
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