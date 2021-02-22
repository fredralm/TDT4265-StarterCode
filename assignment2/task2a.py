import numpy as np
import utils
import typing
from tqdm import tqdm
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean: np.float64, std: np.float64):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    R = np.ones((X.shape[0], X.shape[1]+1), np.float64)
    R[:,:-1] = X
    R = (R - mean)/std
    return R


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    C = np.sum(-(targets*np.log(outputs)))/targets.shape[0]
    return C

def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

def improved_sigmoid(z):
    return 1.7159*np.tanh(2*z/3)

def improved_sigmoid_slope(z):
    return 1.7159*8/(3*np.square(np.exp(-2*z/3)+np.exp(2*z/3)))


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        if use_improved_weight_init:
            # Initialize weights to sampled weights from normal distribution
            for layer_idx, w in enumerate(self.ws):
                self.ws[layer_idx] = np.random.normal(0, 1/np.sqrt(w.shape[0]), size=w.shape)
        else:
            # Initialize weights to random sampled weights between [-1, 1]
            for layer_idx, w in enumerate(self.ws):
                self.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_output = ...
        # Initialize lists for saving intermidiate activation and slope
        self.hidden_layer_output = []
        self.hidden_layer_slope = []
        # Initialize activation
        A = X.T
        self.hidden_layer_output.append(A)
        # Loop over all layers exept output layer and activate forward
        for w in self.ws[:-1]:
            z_j = w.T.dot(A)       # z_j = w_j^T * x
            if self.use_improved_sigmoid:
                A = improved_sigmoid(z_j)
                A_slope = improved_sigmoid_slope(z_j)
            else:
                A = sigmoid(z_j)
                A_slope = A*(1-A)
            # Store activation and slope for backpropagation
            self.hidden_layer_output.append(A)
            self.hidden_layer_slope.append(A_slope)
        # Input to the output layer comes from the final hidden layer, which is the final one to update A
        # Forward pass over output layer with softmax
        z_k = A.T.dot(self.ws[-1])
        Y = np.exp(z_k)
        for i in range(X.shape[0]):
            Y[i] = Y[i]/np.sum(Y[i])
        return Y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        self.grads = []
        # First do backward for output layer
        delta_k = -(targets - outputs)
        grad_k = self.hidden_layer_output[-1].dot(delta_k)/X.shape[0]
        self.grads.append(grad_k)
        # Loop backwards over the hidden layers
        for i in range(len(self.ws)-1, 0, -1):
            # delta_j = f'(z_j)*w_kj^T*delta_k, renamed to delta_k for reuse in backpropagation
            delta_k = (self.hidden_layer_slope[i-1] * (self.ws[i].dot((delta_k.T)))).T
            grad_j = self.hidden_layer_output[i-1].dot(delta_k)/X.shape[0]
            self.grads.append(grad_j)
        # Reverse self.grads to get them in order from input to output
        self.grads.reverse()

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Y_out = np.zeros((Y.shape[0], num_classes), dtype=np.uint8)
    for i in range(Y.shape[0]):
        Y_out[i][Y[i][0]] = 1
    return Y_out


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in tqdm(range(w.shape[0])):
            for j in tqdm(range(w.shape[1])):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"ratio: {model.grads[layer_idx][i, j]/gradient_approximation} "\
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train, X_train.mean(), X_train.std())
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
