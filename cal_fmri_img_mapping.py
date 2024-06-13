import numpy as np
import pandas as pd


# Function to find intervals between non-zero elements
def find_non_zero_intervals(data_array):
    non_zero_indices_start = np.nonzero(data_array)[0]  # Get the indices of non-zero elements
    intervals = []
    non_zero_indices_end = non_zero_indices_start[1:]
    non_zero_indices_end = np.append(non_zero_indices_end, len(data_array))
    for i in range(len(non_zero_indices_start)):
        start = non_zero_indices_start[i]
        end = non_zero_indices_end[i]
        interval = list(range(start, end))
        intervals.append(interval)

    return intervals

    # non_zero_indices = np.nonzero(data_array)[0]  # Get the indices of non-zero elements
    # intervals = []
    #
    # for i in range(len(non_zero_indices) - 1):
    #     start = non_zero_indices[i]
    #     end = non_zero_indices[i + 1]
    #     interval = list(range(start, end))
    #     intervals.append(interval)
    #
    # return intervals


session_id = '01' # from 01 to 40
run_id = '01' # from 01 to 12

file_path = '/scratch/zheng/brain/brain_data_design_matrix/design_session{}_run{}.tsv'.format(session_id, run_id)

df = pd.read_csv(file_path, sep='\t', header=None)

# Convert the DataFrame to a NumPy array
data_array = df.values.flatten()

# Print the NumPy array
print("NumPy array:", data_array)

# Find the indices of non-zero elements
non_zero_indices = np.nonzero(data_array)

# Print the non-zero indices
print("Non-zero indices:", non_zero_indices)
print("Non-zero values:", data_array[non_zero_indices])


# Find the intervals
intervals = find_non_zero_intervals(data_array)

# Print the intervals
for interval in intervals:
    print(f"Indices from {interval[0]} to {interval[-1]}: {interval}")

print(data_array[intervals[0]])

print(data_array[intervals[-1]])

