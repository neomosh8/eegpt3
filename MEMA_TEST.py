import scipy.io

# Replace 'your_file.mat' with the path to your .mat file
mat_data = scipy.io.loadmat('dataset/Subject2_attention_4.mat')

# Print all keys in the .mat file (some keys are added by MATLAB)
print("Keys in the mat file:")
for key in mat_data.keys():
    print(key)

# If you want to see the contents of each key, you can do:
print("\nDetailed content:")
for key, value in mat_data.items():
    print(f"{key}:")
    print(value)
    print("-" * 40)
