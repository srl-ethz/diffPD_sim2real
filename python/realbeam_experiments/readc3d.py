# Read C3D files in Python

import c3d
import cv2
import os
import numpy as np

points = []
reader = c3d.Reader(open('Measurement_data/beam_load52_V2_b.c3d', 'rb'))
for i, point, analog in reader.read_frames():
    points.append(point[:,:3])
    print('frame {}: point {}, analog {}'.format(i, point.shape, analog.shape))

### Motion Capture Points are 5D data: (x,y,z) and then (error_estimate) and (number of cameras that captured the point), point shape is (N, 5) then.

points = np.stack(points)

video_name = 'video.avi'
width, height = (900, 600)

# Pixel coordinates, transform it such that it fits the frame
margin = 0.95   # Percentual decrease of image (introducing borders)
point_coords = np.round(
    height * (margin*points - np.min(points)) / (np.max(points) - np.min(points))
).astype(int)

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 0 Red, 1 Blue, 2 Green, 3 Magenta, 4 Cyan, 5 Yellow, 6 Dark Gray, 7 White, 8 Gray
color = ([0,0,255], [255,0,0], [0,255,0], [255,0,255], [255,255,0], [0,255,255], [50,50,50], [255,255,255], [128,128,128])

frames = []
for i in range(points.shape[0]):
    img = np.zeros((height, width, 3), np.uint8)
    for j in range(points.shape[1]):
        ### Just plotting X and Z coordinates here. Don't forget that width and height are flipped in matrix indexing.
        img = cv2.circle(img, (point_coords[i,j,2], point_coords[i,j,0]), 5, color[j], -1)
    frames.append(img)
    ### If we want to write the whole video
    video.write(img)

i = 0
while i < len(frames):
    cv2.imshow('frame', frames[i])
    key = cv2.waitKey(0) & 0xFF    
    # waitKey(0) pauses until a key is pressed, waitKey(1) checks for keypresses in 1ms intervals, and displays next frame according to video fps
    # I guess 0xFF makes it a 1Byte character?
    # ord() gets the unicode value of a character

    if key == ord('q'):
        # Get out
        break
    elif key == ord('h'):
        # Go backward 10 frames
        i -= 10
    elif key == ord('j'):
        # Go backward one frame
        i -= 1
    elif key == ord('k'):
        # Go forward one frame
        i += 1
    elif key == ord('l'):
        # Go forward 10 frames
        i += 10
    elif key == ord('s'):
        start_frame = i
        print(f"Start frame: {i}")
    elif key == ord('e'):
        end_frame = i
        print(f"End frame: {i}")
    else:
        # Default behaviour: do nothing
        continue


cv2.destroyAllWindows()
video.release()
