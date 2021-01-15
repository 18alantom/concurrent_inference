# concurrent_inference

An example of how to use the `multiprocessing` package along with PyTorch.

Code pertaining to this [Medium post](https://18alan.medium.com/concurrent-inference-e2f438469214).

---

## What does it do?
![Data flow diagram](media/usecase.png)
- A processes [R] reads images from a folder [Fo] and multiple detection processes [D#] are used to obtain the class wise count of objects in the images and write it to a file [Fi].
- It uses `torch.multiprocessing` for multiprocessing, `PIL.Image` to read the images and `tqdm` to keep track of the queue.  
![processing](media/processing.gif)

## Usage
- Basic usage : `$ python count_objects.py -f input_folder -o output_file.log`
- For other options : `$ python count_objects.py -h`
