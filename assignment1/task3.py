import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    correct = 0
    total = targets.shape[0]
    outputs = model.forward(X)
    for i in range(targets.shape[0]):
        if targets[i][np.argmax(outputs[i])] == 1:
            correct += 1
    accuracy = correct/total
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        Y_hat = self.model.forward(X_batch)
        self.model.backward(X_batch, Y_hat, Y_batch)
        self.model.w += -(self.model.grad*self.learning_rate)
        loss = cross_entropy_loss(Y_batch, Y_hat)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model

    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg1, val_history_reg1 = trainer.train(num_epochs)
    w1 = model1.w
    # You can finish the rest of task 4 below this point.
    w0 = model.w
    w1 = model1.w
    # Plotting of softmax weights (Task 4b)
    img0 = np.zeros((28, 280))
    img1 = np.zeros((28, 280))

    plt.gray()

    for k in range(10):
        img0[:,28*k:28*(k+1)] = w0[:-1,k].reshape(28,28)
        img1[:,28*k:28*(k+1)] = w1[:-1,k].reshape(28,28)
        #plt.savefig("task4b_weights_lambda=0.png")
    plt.imshow(img0)
    plt.show()
    plt.imshow(img1)
    plt.show()

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    model2 = SoftmaxModel(l2_reg_lambda=l2_lambdas[1])
    model3 = SoftmaxModel(l2_reg_lambda=l2_lambdas[2])
    model4 = SoftmaxModel(l2_reg_lambda=l2_lambdas[3])
    trainer = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    w2 = model2.w
    train_history_reg2, val_history_reg2 = trainer.train(num_epochs)
    trainer = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    w3 = model3.w
    train_history_reg3, val_history_reg3 = trainer.train(num_epochs)
    trainer = SoftmaxTrainer(
        model4, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    w4 = model4.w
    train_history_reg4, val_history_reg4 = trainer.train(num_epochs)

    plt.ylim([0.7, .95])
    utils.plot_loss(val_history_reg1["accuracy"], "Validation Accuracy lambda=1.0")
    utils.plot_loss(val_history_reg2["accuracy"], "Validation Accuracy lambda=0.1")
    utils.plot_loss(val_history_reg3["accuracy"], "Validation Accuracy lambda=0.01")
    utils.plot_loss(val_history_reg4["accuracy"], "Validation Accuracy lambda=0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.figure()
    plt.plot(l2_lambdas, [np.square(np.linalg.norm(w1)), np.square(np.linalg.norm(w2)), np.square(np.linalg.norm(w3)), np.square(np.linalg.norm(w4))])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4d_l2_reg_norms.png")
