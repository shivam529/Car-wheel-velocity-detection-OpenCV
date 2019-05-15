import cv2
import numpy as np
from copy import deepcopy
import math
temp=None

cap = cv2.VideoCapture('cars_passing_input.mp4')
wheel_count=[]
tire=0 ## flag to check if second tire has been found in the frame
car=1 ## counter for which car number we are at
flag=0 ## flag to check if its current car, is set to one a car is in frame and to zero again after theres no car for a certain period in frame
wheel_position={} ## dictionary of  each wheel positiion at each frame 
frame_count=0
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
out1 = cv2.VideoWriter('outputcar.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),5, (frame_width,frame_height))
label=[]

while(cap.isOpened()):
    frame_count+=1
    wno=1
    ret, frame = cap.read()
    if ret:

        img=frame

        ## unused copy
        img1=deepcopy(img)
        
        ## gray scale the image##
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ## remove noises ##
        blur = cv2.GaussianBlur(gray,(11,11),0)
        ## find edges (to be passed to hough circles) ##
        edges = cv2.Canny(blur,0,200,apertureSize = 3)

        ## Detect circles in the image ##
        circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2,200,
                                        param1=60,param2=25,minRadius=54,maxRadius=77)
    
        circles = np.uint16(np.around(circles))
       
        ## Function used to find speed of the wheel in contention##
        ## For every wheel, we find the minimum position(i.e, starting position) and maximum position right now(ending position) and the 
        ## associated frames with it this way we can computes pixels/frame
       
        def current_speed(cur_wheel):
            s=max(cur_wheel,key=lambda x:x[0][0])
            e=min(cur_wheel,key=lambda x:x[0][0])
            start=s[0][0]*1.0
            end=e[0][0]*1.0
            s_frame=s[1]*1.0
            e_frame=e[1]*1.0
            h=(abs((end-start)/(s[1]-e[1])))
            if(math.isnan(h)):
                return 100
            else:

                return math.ceil(h)
        ## Function for template matchnig to remove wrongly detected circles ##
        
        def ssd(img1):
            try:
                tire1=cv2.imread("wheel977.png",0)
                kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
                dstt= cv2.morphologyEx(tire1, cv2.MORPH_CLOSE, kernel)
                dstt1= cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
                dstt=cv2.resize(dstt,(150,150),interpolation=cv2.INTER_CUBIC)
                dstt1=cv2.resize(dstt1,(150,150),interpolation=cv2.INTER_CUBIC)
                diff=dstt-dstt1
                diff=diff**2
                ssd=np.sum(diff)
                return ssd
            except:
                pass

        if(len(wheel_count)>0):
            last_tire=wheel_count[-1]
            if(last_tire==1):
                flag=1
            if(last_tire==2):
                tire=1
            if(len(wheel_count)>4):
                if(sum(wheel_count[-4:])==0 and flag==1):
                    car+=1
                    tire=0
                    flag=0

        c=0
        detected_wheels=[]
        w=[]
        for i in circles[0,:]:
            mask=np.zeros(img.shape,dtype=np.uint8)
            out=mask*img
            white=255-mask
            k=out+white
            
            rectX = (i[0] - i[2]) 
            rectY = (i[1] - i[2])

            crop_img = gray[rectY:(rectY+2*i[2]), rectX:(rectX+2*i[2])]

            ## Assumptions: tires are in same line, they don't deviate much in vertical direction, this way circles aren't detected outside tire path.
            
            if( i[2]<75 and i[2]>53 and i[1]<560 and i[1]>500): 
                detected_wheels.append(crop_img)
                (thresh, final_box) = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
                try:
                    if(ssd(final_box)<150000):
                        pass
                    else:
                        c+=1
                        w.append((i[0],i[1],i[2]))
                except:
                    pass
                
                ## w contains circles associated with tires
                
                w=sorted(w,key=lambda x:x[0],reverse=True)
    
        t=len(w)    
        
        for x in w:
           
            cv2.circle(img,(x[0],x[1]),x[2],(0,0,255),2)
            if(tire==0):
                cv2.putText(img,str(2*(car)-wno), (x[0],x[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),thickness=3)
                label.append((2*(car)-wno))
                temp=[((x[0],x[1]),frame_count)]
                wheel_position[2*(car)-wno]=wheel_position.get((2*(car)-wno),[])+temp
                speed=current_speed(wheel_position[2*(car)-wno])
                cv2.putText(img,str(speed)+"pixels/frame", (x[0],x[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0),thickness=1)
                wno-=1
            if(tire==1 and t==2):
                cv2.putText(img,str(2*(car)-wno), (x[0],x[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),thickness=3)
                label.append((2*(car)-wno))
                temp=[((x[0],x[1]),frame_count)]
                wheel_position[2*(car)-wno]=wheel_position.get((2*(car)-wno),[])+temp
                speed=current_speed(wheel_position[2*(car)-wno])
                cv2.putText(img,str(speed)+"pixels/frame", (x[0],x[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0),thickness=1)
                wno-=1
            if(tire==1 and t==1):
                cv2.putText(img,str(2*(car)), (x[0],x[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),thickness=3)
                label.append((2*(car)))
                temp=[((x[0],x[1]),frame_count)]
                wheel_position[2*(car)]=wheel_position.get((2*(car)),[])+temp
                speed=current_speed(wheel_position[2*(car)])
                
                cv2.putText(img,str(speed)+"pixels/frame", (x[0],x[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0),thickness=1)
            cv2.putText(img,str(max(label)), (1700,200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255),thickness=3)

        
        wheel_count.append(c)

        cv2.imshow("car",img)
        out1.write(img)
        cv2.waitKey(1)
    else:
        break
    

cap.release()
out1.release()
cv2.destroyAllWindows()



