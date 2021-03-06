import time
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm.auto import tqdm
from read_and_detect import read_images_into_q, detect_objects
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_detector():
    return fasterrcnn_resnet50_fpn(True)

def print_qsize(event, precv_pipe, queue):
    try:
        pbar = tqdm(bar_format="{desc}")
        while not (event.is_set() and queue.empty()):
            if not precv_pipe.poll(): continue
            remaining, name = precv_pipe.recv()
            pbar.desc = f"rem : {remaining:4}, " + \
                f"qsize : {queue.qsize():2}, " + \
                f"current : {name}"
            pbar.update()
            time.sleep(0.05)
        pbar.close()
    except NotImplementedError as err:
        print("JoinableQueue.qsize has not been implemented;"+
            "remainging can't be shown")

def caller(device, images_path, output_path, detector_count=2, qsize=8):
    start = time.time()
    # Initialize sync structures
    queue = mp.JoinableQueue(qsize)
    event = mp.Event()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()

    # Initialize processes
    reader_process = mp.Process(
        target=read_images_into_q,
        args=(images_path, queue, event, psend_pipe)
    )
    detector_processes = [\
            mp.Process(\
                target=detect_objects,\
                args=(queue, event, get_detector(),\
                    device, lock, output_path))\
            for i in range(detector_count)]

    # Starting processes
    reader_process.start()
    [dp.start() for dp in detector_processes]

    print_qsize(event, precv_pipe, queue)

    # Waiting for processes to complete
    [dp.join() for dp in detector_processes]
    reader_process.join()

    # Closing everything
    [c.close() for c in closables]
    print(f"time taken : {time.time() - start} s.")
