from onvif import ONVIFCamera
from time import sleep
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import math
import cv2
from multiprocessing import Process 


def mov_to_face(ptz, request, x, y, to_x, to_y, speed_kof = 1, timeout=0):
	if (x < to_x +40 and x > to_x -40 and y < to_y +40 and y > to_y -40):
		request.Velocity.PanTilt._x = 0
		request.Velocity.PanTilt._y = 0
		ptz.ContinuousMove(request)
	else: 
		len_x = -(to_x - x)
		len_y = (to_y - y)
		vec = math.sqrt(len_x**2+len_y**2)
		vec_x = (len_x/(vec/100.0))/100.0
		vec_y = (len_y/(vec/100.0))/100.0
		print str(vec_x)+" : "+str(vec_y)
		request.Velocity.PanTilt._x = vec_x*speed_kof
		request.Velocity.PanTilt._y = vec_y*speed_kof
		ptz.ContinuousMove(request)
	sleep(timeout)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
head_cascade = cv2.CascadeClassifier('cascadeH5.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')


print 'conection with camera...'
mycam = ONVIFCamera('192.168.1.102', 80, 'admin', 'Supervisor', '/etc/onvif/wsdl/')
#mycam = ONVIFCamera('172.16.83.102:554', 80, 'admin', 'Supervisor')
#mycam = ONVIFCamera('192.168.13.12', 80, 'admin', 'Supervisor')
media = mycam.create_media_service()
profile = media.GetProfiles()[0]
ptz = mycam.create_ptz_service()
request = ptz.create_type('GetConfigurationOptions')
request.ConfigurationToken = profile.PTZConfiguration._token
ptz_configuration_options = ptz.GetConfigurationOptions(request)
request = ptz.create_type('ContinuousMove')
request.ProfileToken = profile._token
print 'sucsess conection.'

print("starting video stream...")
vs = VideoStream(src='rtsp://192.168.1.102:554/Streaming/Channels/101').start()
#vs = VideoStream(src='rtsp://192.168.13.12:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1').start()
#vs = VideoStream(src='rtsp://172.16.83.102:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1').start()

image = vs.read()
final_wide = 700
print (image.shape[0])
print (image.shape[1])
r = float(final_wide) / image.shape[1]
new_shape = (final_wide, int(image.shape[0] * r))
print (new_shape[0])
print (new_shape[1])

#full body
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

time.sleep(1.0)
fps = FPS().start()



while True:
	count = 0
	frame = vs.read()
	''', interpolation = cv2.INTER_AREA'''
	frame = cv2.resize(frame, new_shape)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	print "--"
	
	cv2.rectangle(frame,(new_shape[0]/3 - 30, new_shape[1]/3 -30),
		(new_shape[0]/3 + 30,new_shape[1]/3 + 30),(100,100,100),2)
	cv2.rectangle(frame,(2*new_shape[0]/3 - 30, new_shape[1]/3 -30),
		(2*new_shape[0]/3 + 30,new_shape[1]/3 + 30),(100,100,100),2)
	cv2.rectangle(frame,(new_shape[0]/3 - 30, 2*new_shape[1]/3 -30),
		(new_shape[0]/3 + 30,2*new_shape[1]/3 + 30),(100,100,100),2)
	cv2.rectangle(frame,(2*new_shape[0]/3 - 30, 2*new_shape[1]/3 -30),
		(2*new_shape[0]/3 + 30,2*new_shape[1]/3 + 30),(100,100,100),2)
	'''
	#full body
	found,w=hog.detectMultiScale(gray, winStride=(8,8), padding=(32,32), scale=1.1)
	for (x1,y1,w1,h1) in found:
		count += 1
		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
	'''
	'''
	face = head_cascade.detectMultiScale(gray, 1.2, 6)
	for (x1,y1,w1,h1) in face:
		count += 1
		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)	
	'''
	
	# face
	face = face_cascade.detectMultiScale(gray, 1.1, 5)
	for (x1,y1,w1,h1) in face:
		count += 1
		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)	
	if count == 1:
		mov_to_face(ptz, request, x1+w1/2, y1+h1/2, new_shape[0]/3, new_shape[1]/3, speed_kof=0.5)
	

	if count == 0:
		profile = profile_cascade.detectMultiScale(gray, 1.1, 5)
		for (x1,y1,w1,h1) in profile:
			count += 1
			cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
		if count == 1:
			mov_to_face(ptz, request, x1+w1/2, y1+h1/2, new_shape[0]/3, new_shape[1]/3, speed_kof=0.5)


	if count == 0:
		gray2 = cv2.flip(gray, 1)
		profile = profile_cascade.detectMultiScale(gray2, 1.1, 5)
		for (x1,y1,w1,h1) in profile:
			count += 1
			cv2.rectangle(frame,(new_shape[0]-x1,y1),(new_shape[0]-x1-w1,y1+h1),(100,200,200),2)
		if count == 1:
			mov_to_face(ptz, request, new_shape[0]-x1-w1/2, y1+h1/2, new_shape[0]/3, new_shape[1]/3, speed_kof=0.5)


	if count != 1:
		request.Velocity.PanTilt._x = 0
		request.Velocity.PanTilt._y = 0
		ptz.ContinuousMove(request)
		if count > 1:
			print 'many face on frame'
			

	

	cv2.imshow("Frame", frame)
		
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	fps.update()
ptz.Stop({'ProfileToken': request.ProfileToken})
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
