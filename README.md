# concurrent_inference

An example of how to use the `multiprocessing` package along with PyTorch.

---

## What does it do?
- A processes reads images from an input folder and puts it into a queue.
- Multiple processes running detectors (on the same GPU if present) get images from the queue and write the count of detected objects into an output file.
- It uses `torch.multiprocessing` for multiprocessing, `PIL.Image` to read the images and `tqdm` to keep track of the queue.  
![processing](processing.gif)

## Usage
- Basic usage : `$ python count_objects.py -f input_folder -o output_file.log`
- For other options : `$ python count_objects.py -h`
