import json
import h5py

def print_h5_keys(h5file):
    def visit_fn(name, obj):
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
            print(name)
    h5file.visititems(visit_fn)

def process_config_and_h5_files(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    for key in config:
        print(f"Processing {key}:")
        
        test_y_h5 = config[key].get("test_y_h5")
        test_x_h5 = config[key].get("test_x_h5")
        
        if test_y_h5:
            print(f"  - Reading {test_y_h5} (test_y_h5)")
            with h5py.File(test_y_h5, 'r') as f_y:
                print_h5_keys(f_y)
        
        if test_x_h5:
            print(f"  - Reading {test_x_h5} (test_x_h5)")
            with h5py.File(test_x_h5, 'r') as f_x:
                print_h5_keys(f_x)

config_file = "config.json"
process_config_and_h5_files(config_file)
