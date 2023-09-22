import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import wittgenstein as lw
from torch.utils.data import TensorDataset, DataLoader
from avg.neural.nns import SimpleVAE, BinaryVAE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

"""
Similar to vae_ripper, but uses binary latent features in an effort to learn simpler RIPPER models.
This uses a more complex VAE architecture, since the one from vae_simple.py caused the latent space
to collapse to single binary representation for all instances in the dataset (i.e. for a latent 
dimension of 4, every instance in the original dataset of dimensionality 10 was mapped to (1, 0, 0, 0)). 

This uses a staged joint training approach, where the VAE is trained for a few epochs and then the
joint training begins.

Also, need to see what happens if we use the reconstruction of each instance through the entire VAE pipeline,
not just the encoder. We'll have the same number of features, but maybe we'll still be able to learn simpler rules 
"""


def pre_training(vae, data, optimizer, epochs=5):
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, mean, logvar = vae(data)

        reconstruction_loss, kl_divergence = get_vae_loss(reconstructed, data, mean, logvar)
        vae_loss = reconstruction_loss + kl_divergence

        vae_loss.backward()
        optimizer.step()

        print(f"VAE pre-training Epoch {epoch + 1}/{epochs}, VAE Loss: {vae_loss.item()} "
              f"(MSE: {reconstruction_loss}), KLD: {kl_divergence}")


def prepare_ripper_input(vae, x_train_tensor, x_test_tensor, y_train, y_test):
    """Generates the VAE-reconstructed images of the training and the testing set for RIPPER."""

    with torch.no_grad():
        x_train_latent_representation = vae.encode(x_train_tensor)[0].numpy()
        x_test_latent_representation = vae.encode(x_test_tensor)[0].numpy()

        """
        # Convert the latent representation back to a tensor
        latent_tensor = torch.tensor(latent_representation, dtype=torch.float32)

        # Decode the latent representation to get the reconstructed image/data
        reconstructed_data = vae.decode(latent_tensor)

        # Binarize with thresholding
        reconstructed_data_binary = (reconstructed_data > 0.5).float()

        # Convert the tensor to a numpy array if needed
        reconstructed_data_numpy = reconstructed_data_binary.numpy()

        latent_representation = reconstructed_data_numpy
        """

        # A dataframe with the training instances
        x_train_latent_df = pd.DataFrame(x_train_latent_representation,
                                         columns=[f"lf_{i}" for i in range(x_train_latent_representation.shape[1])])

        # Set the labels column
        x_train_latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string

        # A dataframe with the testing instances
        x_test_latent_df = pd.DataFrame(x_test_latent_representation,
                                        columns=[f"lf_{i}" for i in range(x_test_latent_representation.shape[1])])
        x_test_latent_df['goal_status'] = y_test.values.astype(str)

    return x_train_latent_df, x_test_latent_df


def train_ripper_ovr(train_data, test_data, unique_classes):
    """Training RIPPER in an one-vs-rest fashion"""

    ripper_results = {}

    for cls in unique_classes:
        ripper_clf = lw.RIPPER()

        training_data = train_data.copy()
        training_data['goal_status'] = training_data['goal_status'].apply(lambda x: 1 if x == cls else 0)

        testing_data = test_data.copy()
        testing_data['goal_status'] = testing_data['goal_status'].apply(lambda x: 1 if x == cls else 0)

        if training_data['goal_status'].sum() > 0:
            # Train RIPPER for one class
            ripper_clf.fit(training_data, class_feat='goal_status', pos_class=1)

            # Evaluate on the training data
            train_set_predictions = ripper_clf.predict(training_data)
            # train_set_accuracy = (train_set_predictions == training_data['goal_status']).mean()

            f1_score_train = f1_score(train_set_predictions, training_data['goal_status'])

            # Evaluate on the testing set
            test_set_predictions = ripper_clf.predict(testing_data)
            # test_set_accuracy = (test_set_predictions == testing_data['goal_status']).mean()

            f1_score_test = f1_score(test_set_predictions, testing_data['goal_status'])

            model_size = sum([len(rule) for rule in ripper_clf.ruleset_])

            ripper_results[cls] = (f1_score_train, f1_score_test, model_size, ripper_clf.ruleset_)

    return ripper_results


def get_kl_div(mean, logvar, data_size=1):
    return (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) / data_size


def get_vae_loss(reconstructed_data, original_data, mean, logvar, reduction='mean'):
    if reduction == 'mean':
        recon_loss = F.mse_loss(reconstructed_data, original_data, reduction='mean')
    else:
        recon_loss = F.mse_loss(reconstructed_data, original_data, reduction='sum')

    kld_loss = get_kl_div(mean, logvar, len(original_data))  # Call without the last arg to get a sum loss
    return recon_loss, kld_loss


"""
I need to:
- Give batches during VAE training (see vae_simple.py)
- Implement the staged approach with cycles (train the VAE for a few epochs, then learn a RIPPER classifier),
  not at every epoch
"""


def train_vae_combined(vae, X_train_tensor, X_test_tensor, y_train, y_test, optimizer, epochs=5, alpha=0.1):
    """
    Combines the VAE loss with that of RIPPER's. It uses a staged approach where the VAE is
    trained for a number of epochs (isolated_vae_training), then VAE + RIPPER
    """

    print('Starting joint training')

    unique_classes = y_train.unique()

    for epoch in range(epochs):
        optimizer.zero_grad()

        reconstructed, mean, logvar = vae(X_train_tensor)

        reconstruction_loss, kl_divergence = get_vae_loss(reconstructed, X_train_tensor, mean, logvar)
        vae_loss = reconstruction_loss + kl_divergence

        # Extracting latent representation for RIPPER
        x_train_reconstructed, x_test_reconstructed = prepare_ripper_input(vae, X_train_tensor,
                                                                           X_test_tensor, y_train, y_test)
        print('Training RIPPER...')

        ripper_results = train_ripper_ovr(x_train_reconstructed, x_test_reconstructed, unique_classes)

        avg_train_f1 = sum(entry[0] for entry in ripper_results.values()) / len(ripper_results)
        avg_test_f1 = sum(entry[1] for entry in ripper_results.values()) / len(ripper_results)
        model_size = sum(entry[2] for entry in ripper_results.values())

        ripper_loss = (1 - avg_train_f1) + model_size * alpha
        combined_loss = vae_loss + ripper_loss

        # for name, param in vae.named_parameters():
        #    print("Before backward:", name, param.data)

        combined_loss.backward()
        optimizer.step()

        # for name, param in vae.named_parameters():
        #    print("After step:", name, param.data)

        print(
            f"\nEpoch {epoch}/{epochs}\nVAE -- Loss: {vae_loss.item()} (MSE: {reconstruction_loss} | KLD: {kl_divergence})\n"
            f"RIPPER -- Macro F1 (train, test): ({avg_train_f1}, {avg_test_f1}), Model Size: {model_size},"
            f" Loss: {ripper_loss}\n"
            f"Combined VAE + RIPPER Loss: {combined_loss.item()}\n\nRIPPER Model:")

        for (k, v) in ripper_results.items():
            print(f'Class: {k}, F1 (train | test): ({v[0]} | {v[1]})')
            print(v[3])
            print("-" * 50)

    # Save the final latent representation of the dataset to CSV
    with torch.no_grad():
        final_latent_representation = vae.encode(X_train_tensor)[0].numpy()
        final_latent_df = pd.DataFrame(final_latent_representation,
                                       columns=[f"lf_{i}" for i in range(final_latent_representation.shape[1])])
        final_latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string
        final_latent_df.to_csv(
            "/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_vae_ripper_binary.csv", index=False)


if __name__ == "__main__":

    # Standard in a staged approach. Set this to False to get joint training from the start
    pretraining = True
    pretraining_epochs = 100

    training_epochs = 1000
    with_scaling = True  # If true much larger RIPPER models tend to be learnt.

    data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
    features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']

    # The following is necessary in order to remove a single-instance class from the label set
    # (in this case the class '(unknown)'. Including this class will throw an error like:
    #     ValueError: The least populated class in y has only 1 member, which is too few.
    #     The minimum number of groups for any class cannot be less than 2.

    # Identify the problematic class
    counts = data['goal_status'].value_counts()
    single_instance_class = counts[counts == 1].index[0]

    # Separate out the single instance
    # single_instance_df = data[data['goal_status'] == single_instance_class]
    data_ = data[data['goal_status'] != single_instance_class]

    # Make sure to use the stratify arg to ensure that the distribution of all classes
    # in both the training and the testing is similar to that of the original dataset
    # (so all classes are properly represented in both the training & the testing sets).
    X_train, X_test, y_train, y_test = train_test_split(
        data_[features],
        data_['goal_status'],
        test_size=0.3,
        stratify=data_['goal_status'],
        random_state=42
    )

    if with_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    vae_model = BinaryVAE(input_dim=X_train_tensor.shape[1], hidden_dims=[20, 15], latent_dim=6, binary=True)
    # vae_model = SimpleVAE(input_dim=X_train_tensor.shape[1], hidden_dim=7, latent_dim=6)
    # vae_model = EnhancedVAE(input_dim=X_train_tensor.shape[1], hidden_dims=[64, 128, 256], latent_dim=7, binary=True)

    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)

    if pretraining:
        pre_training(vae_model, X_train_tensor, optimizer, epochs=pretraining_epochs)

    train_vae_combined(vae_model, X_train_tensor, X_test_tensor, y_train, y_test, optimizer, epochs=training_epochs)
