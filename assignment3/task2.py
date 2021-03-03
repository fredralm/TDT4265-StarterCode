import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
import os


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = [32, 64, 128]  # Set number of filters in each conv layer
        self.num_classes = num_classes
        # Define the first convolutional layer
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        # Define the rest of the convolutional layers
        for i in range(1, len(num_filters)):
            self.feature_extractor = nn.Sequential(
                self.feature_extractor,
                nn.Conv2d(
                    in_channels=num_filters[i-1],
                    out_channels=num_filters[i],
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        self.num_output_features = 4*4*128
        # Initialize the first fully connected layer
        # Inputs all extracted features from the convolutional layers
        num_nodes = 64
        # Outputs num_nodes features
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_nodes),
            nn.ReLU()
        )
        # Initialize the final fully connected layer
        # Inputs num_nodes features
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            self.classifier,
            nn.Linear(num_nodes, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    train = True
    if train:
        trainer.train()
    else:
        trainer.load_best_model()
        print("Model loaded")
    create_plots(trainer, "task2")
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    train_loss, train_acc = compute_loss_and_accuracy(
        dataloader_train, model, nn.CrossEntropyLoss()
    )
    validation_loss, validation_acc = compute_loss_and_accuracy(
        dataloader_val, model, nn.CrossEntropyLoss()
    )
    test_loss, test_acc = compute_loss_and_accuracy(
        dataloader_test, model, nn.CrossEntropyLoss()
    )
    print("Final training accuracy:", train_acc)
    print("Final validation accuracy:", validation_acc)
    print("Final test accuracy:", test_acc)
