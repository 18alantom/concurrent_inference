FILTER_THRESHOLD = 0.8
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

get_name = lambda label: COCO_INSTANCE_CATEGORY_NAMES[label]
def filter_output(output, threshold=FILTER_THRESHOLD):
    """
    Filters out detector `outputs` below `threshold`.
    """
    mask = output["scores"] > threshold
    for keys, items in output.items(): 
        output[keys] = items[mask]
        
def get_output_string(path, output):
    """
    Combines `path` and `output` to obtain string having 
    the source of the image and the count of detections.
    """
    string_list = [f"{get_name(label)}:{count}" \
                   for label, count in  zip(*output["labels"].unique(return_counts=True))]
    return path.as_posix() + " :: " + ", ".join(string_list) + "\n"

def handle_output(path, output, lock, file):
    """
    Obtains the output string from `path` and `output` and writes
    to `file` by acquiring a `lock`
    """
    filter_output(output)
    output_string = get_output_string(path, output)
    lock.acquire()
    file.write(output_string); file.flush()
    lock.release()
