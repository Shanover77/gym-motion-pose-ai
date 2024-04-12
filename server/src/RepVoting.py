import numpy as np

class VotedArrayBuilder:
    def __init__(self, array_list):
        self.array_list = array_list
        self.majority_length = None
        self.filtered_arrays = None
        self.processed_arrays = None
        self.voted_array = None

    def determine_majority_length(self):
        lengths = [len(entry['indices']) for entry in self.array_list]
        self.majority_length = max(set(lengths), key=lengths.count)

    def filter_arrays(self):
        self.filtered_arrays = [entry for entry in self.array_list if len(entry['indices']) <= self.majority_length + 2]

    def process_arrays(self):
        self.processed_arrays = []
        for entry in self.filtered_arrays:
            indices = entry['indices']
            diff = self.majority_length - len(indices)

            if diff > 0:
                # Pad with the last value
                indices = np.pad(indices, (0, diff), 'edge')
            elif diff < 0:
                # Truncate to the majority length
                indices = indices[:self.majority_length]
            
            self.processed_arrays.append({'column': entry['column'], 'indices': indices})


    def build_voted_array(self):
        self.voted_array = np.mean([entry['indices'] for entry in self.processed_arrays], axis=0)

    def process(self):
        self.determine_majority_length()
        self.filter_arrays()
        self.process_arrays()
        self.build_voted_array()

    def get_voted_array(self):
        return self.voted_array

# Sample data
# sample_data = [{'column': 'left_arm', 'indices': np.array([7, 166, 332, 403, 518, 574, 640])},
#                {'column': 'right_arm', 'indices': np.array([143, 214, 285, 458, 571, 648])},
#                # ... (other entries)
#                {'column': 'leg_spread', 'indices': np.array([86, 179, 333, 400, 571, 659])}]

# Example usage:
# voted_array_builder = VotedArrayBuilder(sample_data)
# voted_array_builder.process()
# result = voted_array_builder.get_voted_array()
# print(result)
