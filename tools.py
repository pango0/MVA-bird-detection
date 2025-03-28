from PIL import Image, ImageDraw
import os
import torch

def get_available_gpus():
    return [f'cuda:{i}' for i in range(torch.cuda.device_count())]

def get_image_resolution(img_path):
    with Image.open(img_path) as img:
        width, height = img.size

    print(f"Resolution: {width} x {height}")
    return width, height

def get_dir_resolution(directory):
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    
    if not files:
        print("No image files found in the directory.")
        return
    image_path = os.path.join(directory, files[0])
    # Open the image and get resolution
    with Image.open(image_path) as img:
        width, height = img.size

    print(f"Resolution: {width} x {height}")
    return width, height

def get_slice(w, h):
    if w == 15360 and h == 8640:
        slice = 1280
    elif w == 7680 and h == 4320:
        slice = 640
    elif w == 3840 and h == 2160:
        slice = 320
    elif w == 1920 and h == 1080:
        slice = 160
    else :
        slice = 320
    print(f'Tile size: {slice}*{slice}')
    return slice


def draw_bounding_boxes(image, coordinates, label_name, output_path):
    image.resize((1920, 1080), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for coordinate in coordinates:
        draw.rectangle(coordinate, outline="red", width=3)
        draw.text((coordinate[0], coordinate[1]), f"{label_name}", fill="red")
    image.save(output_path, format="JPEG", quality=70)
    print(f"Image with bounding boxes saved as {output_path}")