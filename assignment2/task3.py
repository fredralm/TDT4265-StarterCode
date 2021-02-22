import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 30
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = []
    for i in range(10):
        neurons_per_layer.append(64)
    neurons_per_layer.append(10)
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    if use_momentum:
        learning_rate = .02

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = X_train.mean()
    std = X_train.std()
    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train = False #True -> program trains new models, False -> program loads saved models and plots results
    if train: #train new model
        train_history, val_history = trainer.train(num_epochs)
        np.save('model_weights10', model.ws)
        np.save('train_history10', train_history)
        np.save('val_history10', val_history)
    else: #load saved model
        model.ws = np.load('model_weights10.npy', allow_pickle=True)
        train_history = np.load('train_history10.npy', allow_pickle=True)
        val_history = np.load('val_history10.npy', allow_pickle=True)


    # Model for adding incremental change
    extra_model = True
    if extra_model:
        model_one_layer = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_one_layer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_one_layer, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        if train:
            train_history_one_layer, val_history_one_layer = trainer_one_layer.train(
                num_epochs)
            np.save('model_weights2', model_one_layer.ws)
            np.save('train_history2', train_history_one_layer)
            np.save('val_history2', val_history_one_layer)
        else:
            model_one_layer.ws = np.load('model_weights_improved_sigmoid.npy', allow_pickle=True)
            train_history_one_layer = np.load('train_history_improved_sigmoid.npy', allow_pickle=True)
            val_history_one_layer = np.load('val_history_improved_sigmoid.npy', allow_pickle=True)

    if train:
        exit()

    plt.subplot(1, 2, 1)

    utils.plot_loss(train_history.item().get("loss"),
                    "accuracy eleven layers", npoints_to_average=10)
    utils.plot_loss(val_history.item().get("loss"), "Validation loss eleven layers")
    utils.plot_loss(train_history_one_layer.item().get("loss"),
                    "Training loss two layers", npoints_to_average=10)
    utils.plot_loss(val_history_one_layer.item().get("loss"), "Validation loss two layers")

    plt.ylim([0, .4])
    plt.xlabel("Number of training steps")
    plt.ylabel("Training loss")
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1.001])

    utils.plot_loss(train_history.item().get("accuracy"), "Training eleven layers")
    utils.plot_loss(val_history.item().get("accuracy"), "Validation eleven layers")
    utils.plot_loss(train_history_one_layer.item().get("accuracy"), "Training two layers")
    utils.plot_loss(val_history_one_layer.item().get("accuracy"), "Validation two layers")

    plt.xlabel("Number of training steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4e_multilayer.png")
    plt.show()
