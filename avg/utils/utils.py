import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from avg.neural.nns import Autoencoder, BinaryAutoencoder, BinaryConcreteAutoencoder


# ------ SAX Transformation Implementation ------

def normalize_series(series):
    """Normalize the series to zero mean and unit variance."""
    return (series - series.mean()) / series.std()


def sax_transform_value(value, breakpoints):
    """Transform a single value using Symbolic Aggregate Approximation."""
    for i in range(len(breakpoints)):
        if value < breakpoints[i]:
            return chr(97 + i)
    return chr(97 + len(breakpoints))


def column_sax_transform(column, alphabet_size=10):
    """Transform an entire column (time series) using Symbolic Aggregate Approximation."""
    breakpoints = np.percentile(column, np.linspace(0, 100, alphabet_size + 1)[1:-1])
    normalized_column = normalize_series(column)
    return [sax_transform_value(val, breakpoints) for val in normalized_column]


def plot_original_vs_reconstructed(model, data_tensor, samples_num):
    model.eval()
    with torch.no_grad():
        num_samples_to_display = samples_num
        fixed_samples = data_tensor[:num_samples_to_display].numpy()
        if isinstance(model, Autoencoder):
            reconstructions = model(torch.tensor(fixed_samples))
        elif isinstance(model, BinaryAutoencoder) or isinstance(model, BinaryConcreteAutoencoder):
            reconstructions, _ = model(torch.tensor(fixed_samples))
        else:
            reconstructions, _, _ = model(torch.tensor(fixed_samples))
        reconstructions = reconstructions.numpy()

    # Plotting
    for i in range(num_samples_to_display):
        plt.figure(figsize=(10, 3))

        # Original feature vector
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(len(fixed_samples[i])), fixed_samples[i])
        plt.ylim([-2, 2])
        plt.title(f"Original Sample {i + 1}")

        # Reconstructed feature vector
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(reconstructions[i])), reconstructions[i])
        plt.ylim([-2, 2])
        plt.title(f"Reconstruction {i + 1}")

        plt.tight_layout()
        plt.show()
        plt.close()  # Close the figure after displaying

    model.train()


def extract_sample(dataset, sample_size: float):
    """E.g. sample_size=0.1 means we need 1/10th of the data as a sample"""

    if isinstance(dataset, str):
        data = pd.read_csv(dataset)
    else:
        data = dataset

    # Extract features and labels
    features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
    X = data[features]
    y = data['goal_status']

    # Extract a stratified sample
    _, X_sample, _, y_sample = train_test_split(X, y, test_size=sample_size, stratify=y, random_state=42)

    # Combine the features and labels of the sample
    sample_data = pd.concat([X_sample, y_sample], axis=1)

    # Save the sample data to a new CSV file
    sample_data.to_csv('path_to_save_sample.csv', index=False)


def remove_single_instance_class(dataset_path: str):
    """The following is necessary in order to remove a single-instance class from the label set
       (in this case the class '(unknown)'. Including this class will throw an error like:
       ValueError: The least populated class in y has only 1 member, which is too few.
       The minimum number of groups for any class cannot be less than 2."""

    data = pd.read_csv(dataset_path)

    # Identify the problematic class
    counts = data['goal_status'].value_counts()
    single_instance_class = counts[counts == 1].index[0]

    # Separate out the single instance
    data_ = data[data['goal_status'] != single_instance_class]

    return data_


def pre_process_dataset(input_csv):
    """
    Removes all single class instances from the dataset. This is to avoid the
    "ValueError: The least populated class in y has only 1 member, which is too few.
     The minimum number of groups for any class cannot be less than 2"
    error during the per-class computation of F1-scores.

    Also, generates a new dataset consisting of the relevant features only plus a labels column
    """
    df = pd.read_csv(input_csv)

    # Remove rows where the goal_status appears only once
    counts = df['goal_status'].value_counts()
    classes_to_remove = counts[counts == 1].index.tolist()
    df = df[~df['goal_status'].isin(classes_to_remove)]

    # Define the relevant features
    features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']

    # Generate the new dataset
    new_df = df[features + ['goal_status']]

    # Save to a new CSV
    output_csv = input_csv.replace('.csv', '_clean.csv')
    new_df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to: {output_csv}")


if __name__ == "__main__":
    path = '/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv'
    # data = remove_single_instance_class(path)
    # extract_sample(data, 0.01)
    pre_process_dataset(path)
