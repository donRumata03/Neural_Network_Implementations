from mnist_base_worker import recursive_relative_lsdir, split_path, convert_csv_to_array, show_array_image
import numpy as np
import random
from typing import Tuple


def load_mnist_data(amount : int = None, mnist_location : str = "C:\\Users\\Vova\\PycharmProjects\\Neural_Network_test\\mnist-csv",
                    to_shuffle = True, output_process_info = True, debug=False) -> Tuple[np.array, np.array]:
    """
     Returns tuple of array of labels and array of corresponding images
    """
    if debug:
        output_process_info = True
    buff_all_file_names = recursive_relative_lsdir(mnist_location)
    if output_process_info:
        print("Got mnist file names!")

    if to_shuffle:
        random.shuffle(buff_all_file_names)
        if output_process_info:
            print("Shuffled mnist images!")

    if amount is not None:
        all_file_names = buff_all_file_names[:amount]
    else:
        all_file_names = buff_all_file_names

    labels = np.array([int(split_path(p)[-2]) for p in all_file_names])
    if output_process_info:
        print(f"Labeled mnist file names ({len(labels)} pcs)!")

    if debug:
        s = 0
        for i in range(10):
            print(f"Pictures for number {i}: {list(labels).count(i)}")
            s += list(labels).count(i)
        print("Total:", s)

    # Loading images:
    images = np.zeros((len(labels), 28, 28), dtype=np.float64)
    for i in range(len(all_file_names)):
        images[i] = convert_csv_to_array(all_file_names[i])
        percent = 100 * i / len(all_file_names)
        if random.random() < 0.001:
            print(f"Loading is {round(percent, 3)} % ready...   ", end="\n\r")

    print("Your mnist dataset is ready :) !!!")

    return labels, images



if __name__ == '__main__':
    # convert_dataset_to_csv("mnist_png\\", "mnist-csv\\")
    arr = load_mnist_data(10)
    print(arr[0][0])
    print(arr[1][0].reshape((1, 28 * 28)))
    show_array_image(arr[1][0])