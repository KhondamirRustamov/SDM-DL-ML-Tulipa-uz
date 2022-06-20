import tensorflow as tf

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
]


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, input_shape=[20], activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=METRICS)
    return model
