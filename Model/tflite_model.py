import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

# Create the TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the supported ops to include TensorFlow ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Disable the experimental lowering of tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the converted model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)