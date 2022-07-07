import cv2
import hand_detect
import numpy as np
import jetbot
bot = jetbot.Robot()
cam = jetbot.Camera(width=640, height=640)

def decode_det(obj, width=640, height=640): #width, height of input image
    x1, y1, x2, y2, conf, cat = obj
    cx = (x1 + x2) / 2 #center of object
    cy = (y1 + y2) / 2 #center of object
    w = x2 - x1 #width of object
    h = y2 - y1 #heighth of object
    
    #distance from camera
    dist = width / w
    #pixel offset from center    
    offx = cx - width / 2
    #angle
    ang = np.arctan2(dist, offx)
    
    return cx, cy, w, h, dist, offx, ang
    
power = 0.6
while True:
    frame = cam.value
    det = hand_detect.detect(frame).to('cpu').numpy()
    
    if len(det) == 0:
        print('no object found')
        bot.forward(0)
        continue
    cat = det[0, 5]
    code = decode_det(det[0])
    print(cat)
    
    #left
    if cat == 0:
        bot.left(power)
    elif cat == 1:
        bot.right(power)
    elif cat == 2:
        bot.forward(power)        
    elif cat == 3:
        bot.backward(power)
    else:
        bot.forward(0)
    
    if len(det) == 2:
        if det[0, 5] == 4 and det[1, 5] == 4:
            break
        
bot.forward(0)
print('done.')