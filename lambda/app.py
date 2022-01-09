import base64
import io
import json

import boto3
import numpy as np
import tensorflow as tf
from matplotlib.cm import get_cmap
from PIL import Image
from scipy.ndimage import gaussian_filter

IMAGE_SIZE = [224, 224]
CLASS_LABELS = ["meningioma", "glioma", "pituitary", "notumor"]

model = tf.keras.models.load_model("model", compile=False)
model.build([None] + IMAGE_SIZE + [3])

s3 = boto3.client("s3")


def load_image(bucket, key):
    return Image.open(s3.get_object(Bucket=bucket, Key=key)["Body"]).convert(
        "RGB"
    )


def predict(image):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.per_image_standardization(image)

    with tf.GradientTape() as tape:
        tape.watch(image)
        scores = tf.nn.softmax(model(image[tf.newaxis], training=False))[0]
        loss = tf.reduce_max(scores)
    gradient = tape.gradient(loss, image)
    s_map = tf.reduce_max(gradient, axis=2)

    scores = scores.numpy().tolist()

    s_map = gaussian_filter(s_map, 5)
    s_map = np.ma.masked_less(s_map, np.percentile(s_map, 95), copy=False)
    s_map -= s_map.min()
    s_map *= 1 / s_map.max()
    s_map = get_cmap("jet")(s_map, alpha=0.5)
    s_map *= 255
    s_map = s_map.astype(np.uint8)
    s_map = Image.fromarray(s_map)

    return scores, s_map


def encode_image(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()


def lambda_handler(event, context):
    bucket = event["queryStringParameters"]["bucket"]
    key = event["queryStringParameters"]["key"]

    image = load_image(bucket, key)

    scores, s_map = predict(image)

    image = Image.alpha_composite(
        image.convert("RGBA"), s_map.resize(image.size)
    ).convert("RGB")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "scores": dict(zip(CLASS_LABELS, scores)),
                "image": encode_image(image),
            }
        ),
    }
