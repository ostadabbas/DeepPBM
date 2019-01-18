"""
"""
import cv2 as cv
import numpy as np

vidNumber = 7    # ID of the video it can be (1,2,...,9)
loadPath = 'path_to_data/' \
            + 'Video_' + str('%03d' % vidNumber)
vidName = loadPath +  '/Video_' + str('%03d' % vidNumber) + '.avi'
# load video from the Path
cap = cv.VideoCapture(vidName)
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#length = 20000
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv.CAP_PROP_FPS)
NumChan = 3
BMCvid = np.empty([length, NumChan, height, width])
count = 0
while(cap.isOpened() and count < length):
    ret, frame = cap.read()
    if ret:
        BMCvid[count, :, :, :] = np.transpose(frame, (2, 0, 1))
        cv.imshow('frame', frame)
        count += 1
    else:
        cv.waitKey(1000)
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
np.save((loadPath + '/BMC2012_' + str('%03d' % vidNumber)), BMCvid)
#print(BMCvid.shape())
