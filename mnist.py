import numpy as np
import os


def decode_data(image_path, label_path, num_images, image_size):
    with open(image_path, "rb") as f:
        _ = f.read(16)
        buffer = f.read(image_size * image_size * num_images)

    data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    images = data.reshape(num_images, image_size, image_size)

    with open(label_path, "rb") as f:
        _ = f.read(8)
        buffer = f.read(image_size * image_size * num_images)

    data = np.frombuffer(buffer, dtype=np.uint8)
    labels = data.reshape(num_images, 1)

    zero_indices, *_ = np.where(np.squeeze(labels) == 0.)
    one_indices, *_ = np.where(np.squeeze(labels) == 1.)

    x = np.concatenate([images[zero_indices], images[one_indices]], axis=0)
    y = np.concatenate([labels[zero_indices], labels[one_indices]], axis=0)

    indices = np.random.permutation(y.shape[0])
    x = x[indices]
    y = y[indices]

    num_examples = x.shape[0]

    x = np.reshape(x, (num_examples, image_size * image_size))
    x = np.transpose(x) / 255.
    y = np.transpose(y)
    return x, y


def create_dataset(data_dir):
    num_test_images = 10000
    num_train_images = 60000

    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    x_train, y_train = decode_data(train_images_path, train_labels_path, num_train_images, 28)
    x_test, y_test = decode_data(test_images_path, test_labels_path, num_test_images, 28)
    return x_train, y_train, x_test, y_test
