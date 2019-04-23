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

def revert_normalization(image,means,stds):
    for idx,(mean,std) in enumerate(zip(means,stds)):
        image[idx] = (image[idx] * std) + mean
