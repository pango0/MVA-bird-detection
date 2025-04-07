from PIL import Image, ImageDraw
import os
import torch
import json
import pickle

def get_available_gpus():
    return [f'cuda:{i}' for i in range(torch.cuda.device_count())]

def get_image_resolution(img_path):
    """
    get image resolution
    """
    with Image.open(img_path) as img:
        width, height = img.size

    print(f"Resolution: {width} x {height}")
    return width, height

def get_dir_resolution(directory):
    """
    get resolution of a directory
    """
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
    """
    get sahi tile size according to upscaled resolution
    """
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
    """
    downscale and draw bbox
    """
    image.resize((1920, 1080), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for coordinate in coordinates:
        draw.rectangle(coordinate, outline="red", width=3)
        draw.text((coordinate[0], coordinate[1]), f"{label_name}", fill="red")
    image.save(output_path, format="JPEG", quality=70)
    print(f"Image with bounding boxes saved as {output_path}")
    
def get_dir_dimensions(directory):
    """
    get resolutions of all video subdirectories and write to json file
    """
    dirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    sorted_dir = sorted(dirs, key=lambda x: int(x.split('/')[1]) if x.split('/')[1].isdigit() else -1)
    for dir in sorted_dir[1:]:
        d = dir.split('/')[-1]
        print(f'{dir}: ')
        w, h = get_dir_resolution(dir)
        data = {"width":w, "height": h}
        os.makedirs(f"phase_2/predict/{d}", exist_ok=True)
        json_path = f"phase_2/predict/{d}/image.json"
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4) 

def convert_pickles_to_txt(exp_n):
    pickles_dir = f"runs/predict/exp{exp_n}/pickles"
    annotation_dir = f"runs/predict/exp{exp_n}/annotation"

    # Ensure annotation directory exists
    os.makedirs(annotation_dir, exist_ok=True)

    # Iterate over pickle files
    for filename in sorted(os.listdir(pickles_dir)):
        if filename.endswith(".pickle"):
            pickle_path = os.path.join(pickles_dir, filename)
            txt_filename = filename.replace(".pickle", ".txt")
            txt_path = os.path.join(annotation_dir, txt_filename)

            # Read pickle file
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            
            # Convert data to string and save as text
            with open(txt_path, "w") as f:
                for obj in data:
                    if obj.category.name == 'bird':
                        bbox = obj.bbox
                        score = obj.score.value  # Assuming score has a 'value' attribute
                        f.write(f"{bbox.minx}, {bbox.miny}, {bbox.maxx}, {bbox.maxy}, {score}\n")

            print(f"Converted {filename} to {txt_filename}")

# get_dir_dimensions("swin_results")
