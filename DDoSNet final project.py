import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

total_rows = sum(1 for row in open('all_data2.csv', 'rb'))

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.pretrained = False
        self.finished_training = False
        self.encoder = tf.keras.Sequential([
            layers.SimpleRNN(64, activation="relu", return_sequences=True),
            layers.SimpleRNN(32, activation="relu", return_sequences=True),
            layers.SimpleRNN(16, activation="relu", return_sequences=True),
            layers.SimpleRNN(8, activation="relu", return_sequences=True)])

        self.decoder = tf.keras.Sequential([
            layers.SimpleRNN(16, activation="relu", return_sequences=True),
            layers.SimpleRNN(32, activation="relu", return_sequences=True),
            layers.SimpleRNN(64, activation="relu", return_sequences=True),
            layers.SimpleRNN(79, activation="sigmoid")])  # change to softmax

        self.final = tf.keras.Sequential([
            layers.SimpleRNN(1, activation="sigmoid")
        ])

    def call(self, x):
        decoded = None
        if self.finished_training:
            x = tf.expand_dims(x, -1)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            decoded = tf.expand_dims(decoded, -1)
            final = self.final(decoded)
            return final
        x = tf.expand_dims(x, -1)
        if not self.pretrained:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        final = self.final(x)
        if self.pretrained:
            return final
        return decoded



autoencoder = tf.keras.models.load_model("AnomalyDetectorModel")

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                    loss=tf.keras.losses.CategoricalCrossentropy())
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

file_path = "all_data2.csv"


current_row = 0
limit = int(total_rows * 0.7)
for chunk in pd.read_csv(file_path, low_memory=False, chunksize=100000):
    current_row += 1000000
    print("Current chunk: ", current_row)
    pd.set_option('use_inf_as_na', True)
    chunk = chunk.dropna()
    chunk.head()
    to_drop = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol',
               ' Timestamp',
               'SimillarHTTP', ' Timestamp']
    chunk = chunk.drop(columns=to_drop, axis=1)

    chunk[' Label'] = (chunk[' Label'] != "BENIGN").astype(int)
    raw_data = chunk.values

    chunk.head()

    # The last element contains the labels
    labels = raw_data[:, -1]

    data = raw_data[:, 0:-1]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]
    # pretrain autoencoder

    if current_row < limit:
        autoencoder.pretrained = False
        size_to_train = int(len(normal_train_data) * 0.1)
        pretraining_normal_train_data = normal_train_data[0:size_to_train]
        autoencoder.fit(pretraining_normal_train_data, pretraining_normal_train_data,
                        epochs=1,
                        batch_size=32,
                        validation_data=(test_data, test_data), callbacks=[callback],
                        shuffle=True)

        autoencoder.pretrained = True

        autoencoder.fit(normal_train_data, normal_train_data,
                        epochs=1,
                        batch_size=32,
                        validation_data=(test_data, test_data),
                        shuffle=True)

        my_predictions = autoencoder.predict(data)
        my_predictions = np.rint(my_predictions)

        print("Partial check")
        print("Accuracy = {}".format(accuracy_score(labels, my_predictions)))
        print("Precision = {}".format(precision_score(labels, my_predictions)))
        print("Recall = {}".format(recall_score(labels, my_predictions)))
        print("Saving model. Chunk is currently ", current_row)
        print("The total chunks are: ", total_rows)
        print("70%: ", limit)
        autoencoder.save("AnomalyDetectorModel")
        print("Saving complete.")
    else:
        print("Training complete. Saving model with finished_training = True")
        autoencoder.finished_training = True
        autoencoder.save("AnomalyDetectorModel")
        print("Done.")

        print("Attempting to reconstruct model from save...")
        reconstructed_model = keras.models.load("AnomalyDetectorModel")
        print()
        print("Testing accuracy on unseen data:")
        print("Chunk:", current_row)

        predictions = reconstructed_model.predict(data)
        preds = np.rint(predictions)

        print("Accuracy = {}".format(accuracy_score(labels, preds)))
        print("Precision = {}".format(precision_score(labels, preds)))
        print("Recall = {}".format(recall_score(labels, preds)))
