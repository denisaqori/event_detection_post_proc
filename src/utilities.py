import datetime
import errno
import math
import os
import pickle
import time
from os.path import abspath, dirname, join
from typing import Dict, List

import numpy as np


def save_cpickle(fpath, o):
    f = open(fpath, 'wb')
    pickle.dump(o, f, protocol=4)
    f.close()


def load_cpickle(fpath):
    f = open(fpath, 'rb')
    loaded_obj = pickle.load(f)
    f.close()

    return loaded_obj


# this function calculates timing difference to measure how long running certain parts takes
# it returns the string value of time
def time_since(since):
    now = time.time()
    d = now - since
    m = math.floor(d / 60)
    s = d - m * 60
    return '%dm %ds' % (m, s)


def time_diff(timestamp):
    now = time.time()
    return now - timestamp


def time_s_to_str(sec):
    m = math.floor(sec / 60)
    s = sec - m * 60
    return '%dm %ds' % (m, s)


def time_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def create_dir(root_dir_path, subdir_name_list=None):
    """
    Args:
        root_dir_path:
        subdir_name_list:
    Returns:
    """
    # parent directory
    root_dir = os.path.join(get_root_path("main"), root_dir_path)
    if not os.path.exists(root_dir):
        try:
            os.makedirs(root_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # create subdirectories if there is subdir_name_list passed as a parameter
    if subdir_name_list is not None:
        for subdir_path in subdir_name_list:
            subdir = os.path.join(root_dir, subdir_path)
            if not os.path.exists(subdir):
                try:
                    os.makedirs(subdir)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

    return root_dir


def get_module_path() -> str:
    """
    returns: absolute path of this module
    """
    fpath = abspath(__file__)
    parent_dir = dirname(fpath)
    return parent_dir


def get_root_path(val: str) -> str:
    """
    Returns absolute paths of some important root directories in the project. Assumes project structure will remain
    the same since the paths are dependent on the position of the current module with respect to the others.
    Args:
        val (str): the name of the directory
    Returns:
        path (str): the absolute path to that directory
    """
    if val.lower() == "main":
        # two directories up from the current one (utilities)
        path = join(get_module_path(), "clip_creator.py")
    elif val.lower() == "data":
        path = join(dirname(get_module_path()), "data")
    elif val.lower() == "outputs":
        path = join(dirname(get_module_path()), "outputs")
    elif val.lower() == "src" or val.lower() == "source":
        path = join(dirname(get_module_path()), "src")
    elif val.lower() == "cache":
        path = join(dirname(get_module_path()), "cache")
    else:
        path = None
        print("Root path for " + val + " is set to None...")
    return path


def get_files_in_dir(root_dir_path):
    if root_dir_path is not None:

        img_list = []
        for dirName, subdirList, fileList in os.walk(root_dir_path):
            for fname in fileList:
                fpath = os.path.abspath(os.path.join(dirName, fname))
                img_list.append(fpath)
        return img_list
    else:
        return None


def map_categories(all_categories: List[str]) -> Dict[str, int]:
    """
        Maps categories from a string element to an integer.
    Args:
        all_categories (list): List of unique string category names
    Returns:
        cat (dict): a dictionary mapping a sting to an integer
    """
    # assert uniqueness of list elements
    assert len(all_categories) == len(set(all_categories))
    cat = {}

    all_categories.sort()
    for idx, elem in enumerate(all_categories):
        cat[elem] = idx

    return cat


def convert_categories(category_map: Dict[str, int], categories_subset: List[str]) -> np.ndarray:
    """
    Converts a list of categories from strings to integers based on the internal attribute _cat_mapping.
    Args:
        all_categories: List of string category names of a dataset
        categories_subset:
    Returns:
        converted_categories (list): List of the corresponding integer id of the string categories
    """
    converted_categories = []
    for idx, elem in enumerate(categories_subset):
        assert elem in category_map.keys()
        converted_categories.append(category_map[elem])

    converted_categories = np.array(converted_categories)
    return converted_categories


def find_indices_of_duplicates(ls: List[str]) -> Dict[str, List[int]]:
    """
    Calculates the indices of every unique value of the list.
    Args:
        ls (list): a list of values (strings)
    Returns:
        ordered_dict (dictionary): a dictionary sorted by key, mapping every unique value of the list, to the list
        of the indices at which that value is found in ls.
    """

    # create a set of all list elements to ensure that it has only the unique values of the list
    elem_set = set(ls)
    # dictionary to return
    name_to_indices = {}

    # for each unique list element
    for set_elem in elem_set:
        same_elem = []
        # find the index/indices that belong to it in the original list
        for i, list_elem in enumerate(ls):
            if set_elem == list_elem:
                same_elem.append(i)
        # assign the list of its indices to the list value
        name_to_indices[set_elem] = same_elem

    return name_to_indices


def cleanup(dirname: str, name_part: str) -> None:
    """
    Removes any files that contain the string name_part in the directory dirname.
    Returns: None
    """
    print(f"Deleting any existing files in {dirname} containing the string '{name_part}'.")
    for filename in os.listdir(dirname):
        if name_part in filename:
            full_path_to_remove = join(dirname, filename)

            os.remove(full_path_to_remove)
            print(f"Deleted file {full_path_to_remove}")

    print("Cleanup complete!\n")


if __name__ == "__main__":
    res1 = get_root_path("src")
    res2 = get_root_path("CACHE")
    print(res2)
