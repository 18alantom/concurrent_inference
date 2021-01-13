import time
import torch
import torchvision
from PIL import Image
from queue import Empty
from pathlib import Path
from output_handler import handle_output

COMPLETE = "READING_COMPLETE"
def transform(pil_image):
    # Transforms to apply on the input PIL image
    return torchvision.transforms.functional.to_tensor(pil_image)

def read_images_into_q(images_path, queue, send_pipe, psend_pipe, ext="jpg",\
        wait_time=0.05, transform=transform):
    """
    Reader process, if queue is not full it will read an `ext` image from
    `images_path` and put it onto the `queue` after applying the `transform`, 
    else it will wait for `wait_time` for the  queue to free up.
    
    It uses `send_pipe` to signal downstream processes when all images have 
    been entered into the queue.

    It uses `psend_pipe` for indication.
    """
    image_list = list(Path(images_path).rglob(f"*.{ext}"))
    print(f"processing {len(image_list)} images... ")
    
    while len(image_list) > 0:
        if queue.full():
            time.sleep(wait_time)
            continue
        else:
            image_path = image_list.pop()
            image = Image.open(image_path)
            image = transform(image)
            queue.put((image, image_path))
            psend_pipe.send((len(image_list), image_path.name))
    
    send_pipe.send(COMPLETE)
    queue.join()

def detect_objects(queue, recv_pipe, detector, device, lock, output_path):
    """
    Detector process, Reads a transformed image from the `queue`
    passes it to the detector from `get_detector` and processes the 
    output using `lock`  and `output_path` file for handling the output.
    Uses `pipe` to know if all the images have been written to
    the `queue`.
    """

    file = open(output_path.as_posix(), "a")
    detector.eval().to(device)
    while not (recv_pipe.poll() and queue.empty()):
        try:
            image, image_path = queue.get(block=True, timeout=0.1)
        except Empty:
            continue

        with torch.no_grad():
            image = [image.to(device)]
            output = detector(image)[0]
        queue.task_done()
        handle_output(image_path, output, lock, file)
    file.close()
