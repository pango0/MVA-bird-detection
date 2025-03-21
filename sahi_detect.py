from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from PIL import Image, ImageDraw
import os
import argparse

model_path = "facebook/detr-resnet-101-dc5"
detection_model = AutoDetectionModel.from_pretrained(
    model_type='huggingface',
    model_path=model_path,
    confidence_threshold=0.5,
    image_size=640,
    device="cuda",
)

# === Draw bounding boxes function ===
def draw_bounding_boxes(image, coordinates, label_name, output_path):
    draw = ImageDraw.Draw(image)
    for coordinate in coordinates:
        draw.rectangle(coordinate, outline="red", width=3)
        draw.text((coordinate[0], coordinate[1]), f"{label_name}", fill="red")

    image.save(output_path)
    # print(f"Image with bounding boxes saved as {output_path}")

def run_sahi(image_path):
    image_path = image_path
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.15,
        overlap_width_ratio = 0.15,
    )
    
    object_prediction_list = result.object_prediction_list
    coordinates = []
    for object_prediction in object_prediction_list:
        # Get category name
        category_name = object_prediction.category.name
        if category_name != 'bird':
            continue
        # Get bounding box coordinates
        bbox = object_prediction.bbox
        x1 = bbox.minx
        y1 = bbox.miny
        x2 = bbox.maxx
        y2 = bbox.maxy
        coordinates.append((x1, y1, x2, y2))
        # print(f"Category: {category_name}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
    draw_bounding_boxes(Image.open(image_path).copy(), coordinates, "bird", 'sahi_detection.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bird detection")
    args = parser.parse_args()
    # Add the arguments
    parser.add_argument("--image_path", help="Path to the input image")
    run_sahi(args.image_path)



