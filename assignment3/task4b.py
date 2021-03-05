
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        image: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

indices = [14, 26, 32, 49, 52]

filters_numpy = np.zeros((7, 7*len(indices), 3))
activations_numpy = np.zeros((112, 112*len(indices)))

plt.figure(0, figsize=(20, 8))
for i, index in enumerate(indices):
    filters_numpy[:,7*i:7*(i+1)] = torch_image_to_numpy(first_conv_layer.weight[index])
    activations_numpy[:,112*i:112*(i+1)] = torch_image_to_numpy(activation[0][index])
    plt.subplot(2, len(indices), i+1)
    plt.imshow(filters_numpy[:,7*i:7*(i+1)])
    plt.subplot(2, len(indices), len(indices) + i + 1)
    plt.imshow(activations_numpy[:,112*i:112*(i+1)])
plt.savefig("task4b_plot.png")
plt.show()

activation_last_conv = image
for i, m in enumerate(model.children()):
    activation_last_conv = m(activation_last_conv)
    if i == 7:
        break
print(activation_last_conv.shape)

activations_numpy = np.zeros((7, 7*10))

plt.figure(1, figsize=(20, 8))
for i in range(10):
    activations_numpy[:,7*i:7*(i+1)] = torch_image_to_numpy(activation_last_conv[0][i])
    plt.subplot(1, 10, i+1)
    plt.imshow(activations_numpy[:,7*i:7*(i+1)])
plt.savefig("task4c_plot.png")
plt.show()
