from flask import Flask , request , Response, render_template , redirect
import cv2
import time
import mediapipe as mp 
import numpy as np
import tensorflow as tf  


app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands 

label = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

#cnn model
model = tf.keras.models.load_model("D:\\model\\my_model.h5")

padd = 20

@app.route("/")
def index():
	return render_template('index.html')


@app.route('/camera',methods=['POST','GET'])
def getCamera():

	cap = cv2.VideoCapture(0)
	with mp_hands.Hands(
			model_complexity=0,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as hands:

		while cap.isOpened():
			sucess, image = cap.read()
			time.sleep(0.1)

			if cv2.waitKey(0) & 0xFF == ord('q'):
				break

			if not sucess:
				break
				
			else:

				image.flags.writeable = False
				image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

				results=hands.process(image)
				
				image.flags.writeable=True
				image =image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				if results.multi_hand_landmarks:
					
					#for hand_landmarks in results.multi_hand_landmarks:
					for num,hand  in enumerate(results.multi_hand_landmarks):
						x,y,w,h = calc_bounding_rect(image , hand)
						myHand = {}
						## lmList
						mylmList = []
						xList = []
						yList = []
						for id, lm in enumerate(hand.landmark):
							px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
							mylmList.append([px, py, pz])
							xList.append(px)
							yList.append(py)

						## bbox
						xmin, xmax = min(xList), max(xList)
						ymin, ymax = min(yList), max(yList)
						boxW, boxH = xmax - xmin, ymax - ymin
						bbox = xmin, ymin, boxW, boxH
						cx, cy = bbox[0] + (bbox[2] // 2), \
								 bbox[1] + (bbox[3] // 2)

						myHand["lmList"] = mylmList
						myHand["bbox"] = bbox
						myHand["center"] = (cx, cy)
						X , Y , H, W = myHand["bbox"]
						
						cv2.rectangle(image,(x,y),(H+W,W+Y),(250,0,0),2 )
						
						try:
							# croping hands images
							croped_img1 = image[Y - padd:Y+W + padd,X - padd:X+H + padd]
							#reshaping the image to model shape.
							s_near = (cv2.cvtColor(cv2.resize(croped_img1, (28, 28)), cv2.COLOR_RGB2GRAY)) / 255.0
							#reshaping the array.
							img2 = s_near.reshape(-1, 28, 28, 1)
							#pred the output with model.
							model_out = model.predict([img2])[0]
							prd = (np.argmax(model_out))
							print("pred is " ,prd)
							prd =label[prd]
							cv2.putText(image,prd, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

						except:
							cv2.putText(image,"hand not in range", (0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

						mp_drawing.draw_landmarks(
							image,
							hand,
							mp_hands.HAND_CONNECTIONS,
							mp_drawing_styles.get_default_hand_landmarks_style(),
							mp_drawing_styles.get_default_hand_connections_style())
						
				ret , buffer = cv2.imencode('.jpg',image)
				frame = buffer.tobytes()
				yield(b'--frame\r\n'
					b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

		cap.release()
		cv2.destroyAllWindows()
				
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]



@app.route("/video" )
def video():
	return Response(getCamera(),
		mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/start" , methods=['POST', 'GET'])
def start():
	return render_template('video.html')


@app.route("/stop", methods=['POST'])
def stop():

	return redirect("/")
	
if __name__==('__main__'):
	app.run(debug=False)

