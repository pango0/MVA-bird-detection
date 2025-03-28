from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
from PIL import Image
import threading
from queue import Queue
import os
import argparse
from PIL import Image
import glob

from tools import get_image_resolution, get_slice, get_available_gpus, get_dir_resolution, draw_bounding_boxes

def get_model(model_path="facebook/detr-resnet-101-dc5"):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='huggingface',
        model_path=model_path,
        confidence_threshold=0.5,
        image_size=640,
        device="cuda",
    )
    return detection_model

def run_sahi(image_path, model):
    width, height = get_image_resolution(image_path)
    slice = get_slice(width, height)
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height = slice,
        slice_width = slice,
        overlap_height_ratio = 0.05,
        overlap_width_ratio = 0.05,
    )
    object_prediction_list = result.object_prediction_list
    coordinates = []
    for object_prediction in object_prediction_list:
        # Get category name
        category_name = object_prediction.category.name
        if category_name != 'bird':
            continue

        bbox = object_prediction.bbox
        x1 = bbox.minx
        y1 = bbox.miny
        x2 = bbox.maxx
        y2 = bbox.maxy
        
        coordinates.append((x1, y1, x2, y2))
    draw_bounding_boxes(Image.open(image_path).copy(), coordinates, "bird", 'sahi_detection.png')
    print('Done!')
    return coordinates

def run_batch_sahi(directory, device, model_path="facebook/detr-resnet-101-dc5"):
    width, height = get_dir_resolution(directory)
    slice = get_slice(width, height)
    predict(
        model_type='huggingface',
        model_path=model_path,
        model_device=device,
        model_confidence_threshold=0.5,
        source=directory,
        slice_height=slice,
        slice_width=slice,
        overlap_height_ratio=0.05,
        overlap_width_ratio=0.05,
        export_pickle=True
    )
    print(f'Processing done for {directory} on {device}')

def worker(queue, devices, workers_per_gpu):
    gpu_usage = {device: 0 for device in devices}  # Track GPU usage
    while not queue.empty():
        directory = queue.get()
        available_device = None
        
        while available_device is None:
            for device in devices:
                if gpu_usage[device] < workers_per_gpu:
                    available_device = device
                    gpu_usage[device] += 1
                    break
        
        try:
            run_batch_sahi(directory, available_device)
        finally:
            gpu_usage[available_device] -= 1  # Release GPU slot
        queue.task_done()
    
def run_all(directory='pub_test', workers_per_gpu=2):
    dirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for dir in dirs:
        # pub_test/0001
        img_dir = dir.split('/')[1]
        seq_name = f'{dir}/seqinfo.ini'
        jpg_files = sorted(glob.glob(os.path.join(dir, '*.jpg')))
        with open(seq_name, 'w') as f:
            paragraph = f"""
            [Sequence]
            name={dir}
            imDir={img_dir}
            frameRate=24
            seqLength={len(jpg_files)}
            imWidth=3840
            imHeight=2160
            imExt=.jpg
            """
            f.write(paragraph)
            
    queue = Queue()
    devices = get_available_gpus()
    
    for dir in dirs:
        queue.put(dir)
    
    threads = []
    total_threads = len(devices) * workers_per_gpu
    for _ in range(total_threads):
        thread = threading.Thread(target=worker, args=(queue, devices, workers_per_gpu))
        thread.start()
        threads.append(thread)
    
    queue.join()
    for thread in threads:
        thread.join()
    print("All directories processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bird detection")
    parser.add_argument("--path", help="Path to the input image or pub_test")
    parser.add_argument("--action", help="Action to perform: single inference or batch inference", default="all", choices=["all", "single"])
    parser.add_argument("--workers_per_gpu", type=int, default=4, help="Number of workers per GPU")
    args = parser.parse_args()
    model = get_model()
    if args.action=="all":
        run_all(args.path, args.workers_per_gpu)
    if args.action=="single":
        run_sahi(args.path, model)