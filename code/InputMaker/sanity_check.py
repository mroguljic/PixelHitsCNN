import h5py

# Open the HDF5 file in read mode
file_path = '/ssd-data1/mrogul/clusters/v2/L1F_test_x_1d.hdf5'
with h5py.File(file_path, 'r') as f:
    # List all the keys in the file
    keys = list(f.keys())
    print("Keys in the HDF5 file:")
    
    # For each key, print the corresponding shape
    for key in keys:
        dataset = f[key]
        print(f"Key: {key}, Shape: {dataset.shape}")
