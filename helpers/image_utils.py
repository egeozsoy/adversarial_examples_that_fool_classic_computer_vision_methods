import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_result(image, adversarial):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(image[0])  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial[0])  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = adversarial[0] - image[0]
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')
    plt.savefig('plot.png')

    plt.show()


def revert_normalization(image, means, stds):
    for idx, (mean, std) in enumerate(zip(means, stds)):
        image[idx] = (image[idx] * std) + mean


def show_image(image, label='image'):
    cv2.imshow(label, np.uint8(image))
    cv2.waitKey(0)


# this requires extra color conversion
def show_cifar_10_image(image, label='image'):
    img = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    show_image(img, label)
