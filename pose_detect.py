import tensorflow as tf
import numpy as np
import cv2
import os

model_path = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]



from enum import Enum
import math

class Part(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


connections = [
    (Part.LEFT_EAR, Part.LEFT_EYE ),
    (Part.LEFT_EYE, Part.NOSE ),
    (Part.NOSE, Part.RIGHT_EYE ),
    (Part.RIGHT_EYE, Part.RIGHT_EAR ),

    (Part.LEFT_ELBOW, Part.LEFT_WRIST ),
    (Part.LEFT_ELBOW, Part.LEFT_SHOULDER ),
    (Part.LEFT_HIP, Part.LEFT_SHOULDER ),
    (Part.LEFT_HIP, Part.LEFT_KNEE ),
    (Part.LEFT_KNEE, Part.LEFT_ANKLE ),

    (Part.RIGHT_HIP, Part.RIGHT_SHOULDER ),
    (Part.RIGHT_ELBOW, Part.RIGHT_SHOULDER ),
    (Part.RIGHT_ELBOW, Part.RIGHT_WRIST ),
    (Part.RIGHT_HIP, Part.RIGHT_KNEE ),
    (Part.RIGHT_KNEE, Part.RIGHT_ANKLE ),

    (Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER ),
    (Part.LEFT_HIP, Part.RIGHT_HIP)
]

colors = [
    (0, 255, 255),
    (0, 255, 255),
    (0, 255, 255),
    (0, 255, 255),
    
    (0, 0, 255),
    (0, 64, 255),
    (0, 128, 255),
    (64, 64, 255),
    (128, 0, 255),

    (255, 0, 0),
    (255, 64, 0),
    (255, 128, 0),
    (255, 64, 64),
    (255, 0, 128),

    (128, 255, 0),
    (0, 255, 128)
]

def sigmoid(x):
	return (1 / (1 + math.exp(-x)))

def draw_a_pose(image, points, conf, th = 0.5):
    for con, color in zip(connections, colors):
        c0 = conf[con[0].value]
        c1 = conf[con[1].value]
        if c0 < th or c1 < th:
            color = (0, 0, 0)
        p0 = points[con[0].value]
        p1 = points[con[1].value]
        cv2.line(image, p0, p1, color, thickness=2)



def detect(image):
    input_image = cv2.resize(image, (input_width, input_height))
    input_image = np.float32(input_image)
    input_image = (input_image / 127.5) - 1
    input_image = input_image[None, :, :, [2, 1, 0]]
    
    interpreter.set_tensor(input_details[0]['index'], input_image)
    # Run the calculations
    interpreter.invoke()
    # Extract output data from the interpreter
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    offsets = interpreter.get_tensor(output_details[1]['index'])[0]
    
    dh = scores.shape[0]
    dw = scores.shape[1]
    dp = scores.shape[2]
    scores_reshape = scores.reshape(dh * dw, -1)
    max_index = scores_reshape.argmax(0)
    max_index
    arg_y, arg_x = np.unravel_index(max_index, (dh, dw))
    off_y = offsets[arg_y, arg_x, np.arange(dp)]
    off_x = offsets[arg_y, arg_x, np.arange(dp, dp * 2)]
    conf = scores[arg_y, arg_x, np.arange(dp)]
    #sigmoid
    conf = 1 / (1 + np.exp(-conf))

    point_x = (arg_x / (dw - 1) * input_width + off_x) / input_width * image.shape[1]
    point_y = (arg_y / (dh - 1) * input_height + off_y) / input_height * image.shape[0]

    points = [(int(x), int(y)) for x, y in zip(point_x, point_y)]
    
    dst = image.copy()
    draw_a_pose(dst, points, conf)
    
    return dst
    