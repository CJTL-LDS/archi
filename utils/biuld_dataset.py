import argparse
import os
import random
from PIL import Image
from tqdm import tqdm

def check_overlap(rect1, rect2):
    """Check if two rectangles overlap."""
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])

def create_composite_image(image_paths, num_pngs, save_path, canvas_size=(3600, 3600)):
    """
    Creates a composite image by placing a random number of PNGs onto a white canvas without overlapping.

    Args:
        image_paths (list): A list of paths to the PNG images.
        num_pngs (int): The number of PNGs to place on the canvas.
        save_path (str): The path to save the composite image.
        canvas_size (tuple): The size (width, height) of the canvas in pixels.
    """
    # Create a white canvas
    background = Image.new('RGBA', canvas_size, (255, 255, 255, 255))
    placed_rects = []

    # Select a random subset of images
    selected_images = random.sample(image_paths, min(num_pngs, len(image_paths)))

    for image_path in selected_images:
        try:
            # Open the image and ensure it has an alpha channel
            png = Image.open(image_path).convert("RGBA")

            # Resize image if it's larger than the canvas
            if png.width > canvas_size[0] or png.height > canvas_size[1]:
                png = png.resize((min(png.width, canvas_size[0]), min(png.height, canvas_size[1])), Image.Resampling.LANCZOS)

            max_x = canvas_size[0] - png.width
            max_y = canvas_size[1] - png.height

            # Try to find a non-overlapping position
            for _ in range(100): # Try 100 times
                rand_x = random.randint(0, max_x)
                rand_y = random.randint(0, max_y)
                new_rect = (rand_x, rand_y, rand_x + png.width, rand_y + png.height)

                if not any(check_overlap(new_rect, r) for r in placed_rects):
                    # Paste the PNG onto the background
                    background.paste(png, (rand_x, rand_y), png)
                    placed_rects.append(new_rect)
                    break
            else:
                print(f"Warning: Could not find a non-overlapping position for {image_path}. Skipping.")

        except Exception as e:
            print(f"Could not process image {image_path}: {e}")

    # Save the resulting image
    background.convert("RGB").save(save_path, "PNG")

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of composite images.")
    parser.add_argument("--num_images", type=int, default=32, help="Number of composite images to generate.")
    parser.add_argument("--num_pngs", type=int, default=5, help="Number of PNGs to place on each canvas.")
    parser.add_argument("--save_path", type=str, default="./dataset", help="Directory to save the generated images.")
    parser.add_argument("--png_folder", type=str, default="./png", help="Folder containing the source PNG files.")
    
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Get all png file paths
    try:
        image_files = [os.path.join(args.png_folder, f) for f in os.listdir(args.png_folder) if f.endswith('.png')]
        if not image_files:
            print(f"Error: No PNG files found in the '{args.png_folder}' directory.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{args.png_folder}' was not found.")
        return

    print(f"Generating {args.num_images} images, each with {args.num_pngs} PNGs, and saving to '{args.save_path}'...")

    # Generate images
    for i in tqdm(range(args.num_images)):
        output_path = os.path.join(args.save_path, f"img_{i+1}.png")
        create_composite_image(image_files, args.num_pngs, output_path)

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
