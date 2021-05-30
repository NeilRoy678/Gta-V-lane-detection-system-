from types import coroutine
from PIL import ImageGrab
import PIL
from PIL.Image import new
import numpy as np
import cv2
from numpy.core.arrayprint import FloatingFormat
from numpy.core.defchararray import array
from numpy.core.fromnumeric import reshape, size
from numpy.core.function_base import _linspace_dispatcher
from numpy.core.numerictypes import find_common_type
import matplotlib.pyplot  as plt
from numpy import vstack,ones
from numpy.linalg import lstsq
from statistics import mean

slope = 0

def reigion(img,vertices):
    
    mask = np.zeros_like(img)
    match = 255

    cv2.fillPoly(mask,vertices,match)

    masked_image = cv2.bitwise_and(img,mask)

    return masked_image
def d_line(img,lines):
#    img = np.copy(img)
#    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    try:
        ys  = []
        for i in lines:
            for ii in i:
                ys+= [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 800
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m
                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])
        
        final_lanes = {}
        
        for idx in line_dict:
                final_lanes_copy = final_lanes.copy()
                m = line_dict[idx][0]
                b = line_dict[idx][1]
                line = line_dict[idx][2]
                
                if len(final_lanes) == 0:
                    final_lanes[m] = [ [m,b,line] ]
                    
                else:
                    found_copy = False

                    for other_ms in final_lanes_copy:
                        if not found_copy:
                            if abs(other_ms*1) > abs(m) > abs(other_ms*0.9):
                                if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                    final_lanes[other_ms].append([m,b,line])
                                    found_copy = True
                                    break
                            else:
                                final_lanes[m] = [ [m,b,line] ]
        
        line_counter = {}

        for lane in final_lanes:
            line_counter[lane] = len(final_lanes[lane])
        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except:
        print("TRYING")
    ''' 
    try:    
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),thickness= 3)
    except:
        print("TRYING")
    img = cv2.addWeighted(img,0.8,line_img,1,0.0)
    return img
    '''
#cap = cv2.VideoCapture('TEST.mp4')
while True:
    #ret,frame = cap.read()
    frame = np.array(ImageGrab.grab(bbox = (0,40,800,640))) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    #frame = cv2.imread("road.png")
    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    width = frame.shape[1]
    height = frame.shape[0]

    interest_vertices = [
    (0,height-100),
    (0,350),

    ((800),(350)),

    (width,height-100) 
    ]

    ret,frame2 = cv2.threshold(frame1,127,255,0)

    canny = cv2.Canny(frame1,100,200)
    canny = cv2.GaussianBlur(canny,(5,5),0)
    cropped = reigion(canny,np.array([interest_vertices],np.int32))
 
    lines = cv2.HoughLinesP(cropped,rho = 1,theta= np.pi/180,threshold=180,lines=np.array([]),minLineLength=50,maxLineGap=5)
    try:
        l1,l2 = d_line(frame,lines)
        cv2.line(frame, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 10)
        cv2.line(frame, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0],10)


    except:W
        print("TRYING")
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    