import json
import os
import sys
import time
import traceback

from primitive_perceptron import *
from CNN import *
from PIL import Image


def save_as_csv(data : np.array, filename : str):
    for_str = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for_str.append(str(data[i][j]))
            if j != data.shape[1] - 1:
                for_str.append(";")
        for_str.append("\n")

    file = open(filename, "w")
    file.write("".join(for_str))
    file.close()


def convert_image_to_csv(input_path : str, output_path : str):
    img = Image.open(input_path)
    data = np.zeros((img.size[1], img.size[0]))
    for i in range(img.size[1]):
        for j in range(img.size[0]):
            data[i][j] = img.getpixel((j, i))
    save_as_csv(data, output_path)

def convert_csv_to_array(input_path : str) -> np.array:
    lines = open(input_path, "r").read().strip().split("\n")
    data = np.zeros((len(lines), len(lines[0].strip().split(';'))))
    for line_index, l in enumerate(lines):
        splitted_line = l.strip().split(";")
        for j in range(len(splitted_line)):
            data[line_index][j] = float(splitted_line[j])

    return data

def show_array_image(data : np.array):
    img = Image.new(mode="RGB", size=(data.shape[1], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i][j])
            img.putpixel((j, i), (val, val, val))
    img.show()

def save_array_as_image(data : np.array, output_path : str):
    img = Image.new(mode="RGB", size=(data.shape[1], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i][j])
            img.putpixel((j, i), (val, val, val))
    img.save(output_path)

def convert_csv_to_image(input_path : str, output_path : str):
    data = convert_csv_to_array(input_path)
    save_array_as_image(data, output_path)


def show_csv_image(input_path : str):
    data = convert_csv_to_array(input_path)
    show_array_image(data)

def print_as_json(data):
    string = json.dumps(data, ensure_ascii=False, indent=4)
    print(string)

def replace_all(string : str, chars : set, new_char : str = ""):
    res = []
    for this_char in string:
        if this_char in chars:
            res.append(new_char)
        else:
            res.append(this_char)
    return "".join(res)

# Splitting
def all_split(data : str, splitters : set) -> list:
    return split_if(data, lambda x: x in splitters)

def split_words(data : str):
    return split_if(data, lambda x: not x.isalnum())

def split_if(data : str, function) -> list:
    lst_dat = list(data)
    new_lstdat = []
    for char in lst_dat:
        if function(char):
            new_lstdat.append("♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥")
        else:
            new_lstdat.append(char)
    return "".join(new_lstdat).split("♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥")

def mkdirs(path : str) -> str:
    splitted = all_split(path, {"/", "\\"})
    to_add = False
    if ":" in splitted[0]:
        to_add = True

    this_path : str = splitted[0]
    del splitted[0]
    if to_add:
        this_path += "\\"

    while splitted:
        if splitted[0] not in os.listdir(this_path):
            # print(splitted[0], this_path, os.listdir(this_path))
            this_path += splitted[0] if this_path.endswith("\\") or this_path.endswith("/") else "\\" + splitted[0]
            if len(splitted) != 1:
                this_path += "\\"
            try:
                os.mkdir(this_path)
            except Exception as e:
                ex_info = sys.exc_info()
                traceback.print_exception(*ex_info)
                print(e)
        else:

            this_path = os.path.join(this_path, splitted[0])
        del splitted[0]
    return path

def make_all_dirs(dirs : list):
    for directory in dirs:
        mkdirs(directory)

def recursive_relative_lsdir(base_path : str) -> list:
    res = []
    for address, dirs, files in os.walk(base_path):
        for file in files:
            res.append(address + '\\' + file)
    return res

def copy_dir_structure(path0 : str, path1 : str):
    for address, dirs, files in os.walk(path0):
        this_dir = address.split("\\")[1:]
        if this_dir and this_dir[0]:
            string_dir = "/".join([path1.replace("\\", "")] + this_dir)
            mkdirs(string_dir)

def split_path(path : str):
    return split_if(path, lambda x: x == "\\" or x == "/")

def convert_dataset_to_csv(base_directory : str, out_directory : str):
    copy_dir_structure(base_directory, out_directory)
    all_files = recursive_relative_lsdir(base_directory)
    for index, file in enumerate(all_files):
        new_fname_temp = split_path(file)

        new_fname_temp[0] = out_directory
        new_fname_temp[-1] = ".".join(new_fname_temp[-1].split(".")[:-1] + ["csv"])

        new_fname = os.path.join(*new_fname_temp)
        convert_image_to_csv(file, new_fname)

        if random.random() > 0.001:
            print(100 * (index + 1) / len(all_files), "%")

