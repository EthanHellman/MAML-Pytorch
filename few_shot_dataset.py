import random
from torch.utils.data import Subset

def precompute_class_indices(dataset):
    print("pre-computing labels")
    class_indices = {}
    for idx, (_, _, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    print(class_indices)
    return class_indices

def create_few_shot_dataset(dataset, shots, num_classes, class_indices):
    print("creating few-shot dataset")
    few_shot_dataset_indices = []
    for i in range(num_classes):
        few_shot_dataset_indices.extend(random.sample(class_indices[i], min(shots, len(class_indices[i]))))
    return Subset(dataset, few_shot_dataset_indices)
