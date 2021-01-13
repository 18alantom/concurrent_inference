import torch
import argparse
import torch.multiprocessing as mp
from caller import caller
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Get the count of detected objects.")
    parser.add_argument("-f", "--folder", 
            help="folder having the images", type=Path,
            required=True)
    parser.add_argument("-o", "--output",
            help="path of the output file", type=Path,
            required=True, dest="output")
    parser.add_argument("-d", "--detectors", 
            help="number of detector processes", type=int, default=2)
    parser.add_argument("-q", "--qsize", 
            help="size of the image queue", type=int, default=8)
    args = parser.parse_args()

    mp.set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caller(device, args.folder, args.output, args.detectors, args.qsize)
