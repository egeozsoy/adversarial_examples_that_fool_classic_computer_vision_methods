import cv2
import numpy as np
from configurations import dataset_name,model_name,feature_extractor_name,targeted_attack


def plot_result(image, adversarial,save_name,reference_image=None):
    import matplotlib.pyplot as plt
    image = image / 255
    adversarial = adversarial / 255

    plt.figure()

    if reference_image is not None:
        reference_image = reference_image / 255
        plt.subplot(1, 4, 1)
        plt.title('Starting Image')
        plt.imshow(reference_image)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Original')
        plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Adversarial')
        plt.imshow(adversarial)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Difference')
        difference = adversarial - image
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

    else:

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(adversarial)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

    targeted_str: str = 'targeted' if targeted_attack else 'untargeted'
    plt.savefig(save_name)

    plt.close()


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
