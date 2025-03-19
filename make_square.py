import os
import argparse
from PIL import Image
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
# from relighting.image_processor import pil_square_image
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def pil_square_image(image, desired_size = (512,512), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    # Don't resize if already desired size (Avoid aliasing problem)
    if image.size == desired_size:
        return image
    
    # Calculate the scale factor
    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    # Resize the image
    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    # Create a new blank image with the desired size and black border
    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    # Paste the resized image onto the new image, centered
    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))
    
    return new_image

def process_image(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    if not os.path.isfile(input_path):
        return

    try:
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):

            # Open image using PIL
            img = Image.open(input_path).convert("RGB")

            # Apply the extract_ball function
            result = pil_square_image(img, (1024, 1024))

            # Save the processed image
            output_path = os.path.join(output_dir, filename)
            result.save(output_path)
            # print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(input_path)
        # raise(e)
        # pass
        # print(f"Failed to process {filename}: {e}")

def process_images(input_dir, output_dir, num_workers=4):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all filenames in the input directory
    filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Use multiprocessing pool to process images in parallel
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(partial(process_image, input_dir=input_dir, output_dir=output_dir), filenames), total=len(filenames)))

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process images using extract_ball function.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save processed images.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers.")

    args = parser.parse_args()

    # Process images
    process_images(args.input_dir, args.output_dir, args.num_workers)

if __name__ == "__main__":
    main()
