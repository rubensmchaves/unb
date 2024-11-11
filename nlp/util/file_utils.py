import os
import random
from typing import List, Tuple


def get_file_names(folder_name):
    # List to store file names
    file_names = []
    
    # Check if the provided path is a directory
    if os.path.isdir(folder_name):
        # Iterate over all items in the folder
        for file in os.listdir(folder_name):
            # Full path to the item
            full_path = os.path.join(folder_name, file)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(full_path):
                file_names.append(file)
    else:
        raise ValueError(f"The provided folder name '{folder_name}' is not a valid directory.")
    
    return sorted(file_names)


def train_test_split(filenames: List[str], test_size: float) -> Tuple[List[str], List[str]]:
    # Ensure test_size is a valid percentage
    if not 0 < test_size < 1:
        raise ValueError("test_size should be a float between 0 and 1.")
    
    # Shuffle the list of filenames for random splitting
    files = filenames.copy()
    random.shuffle(files)
    
    # Calculate the size of the test set
    test_set_size = int(len(files) * test_size)
    
    # Split the 'files' into test and train sets
    filenames_test = files[:test_set_size]
    filenames_train = files[test_set_size:]
    
    return filenames_train, filenames_test


if __name__ == '__main__':
    n_samples = 5
    print("Test function 'get_file_names':")
    folder = "../corpus"
    files = get_file_names(folder)
    print(files[:n_samples], "...", "\n")  # Outputs the list of filenames in the specified folder
    
    print("Test function 'train_test_split':")
    train_set, test_set = train_test_split(files, test_size=0.3)
    print(f"Files set (samples): {files[:n_samples]}... ({n_samples} of {len(files)})")
    print(f"Train Set (samples): {train_set[:n_samples]}... ({n_samples} of {len(train_set)})")
    print(f"Test Set (samples): {test_set[:n_samples]}... ({n_samples} of {len(test_set)})")    
