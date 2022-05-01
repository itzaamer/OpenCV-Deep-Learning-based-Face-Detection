import cv2
def face_detection(video,dnn_model):
    while True:
        # Capture frame-by-frame
        check,frame = video_cascade.read()
        # Get the height and width of the input image.
        image_height, image_width, _ = frame.shape
        # Perform the required pre-processings on the image and create a 4D blob from image.
        # Resize the image and apply mean subtraction to its channels
        preprocessed_input_image = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),scalefactor=1.0,size=(300,300),mean=(104.0, 177.0, 123.0))

        #passing blob through the network to detect and pridiction
        opencv_dnn_model.setInput(preprocessed_input_image)
        #model.forward() function to get an array containing the bounding boxes coordinates normalized to ([0.0, 1.0]) and the detection confidence of each faces in the image.
        detections = opencv_dnn_model.forward()
        # Loop through each face detected in the image.
        for face in detections[0][0]:        
            # Retrieve the face detection confidence score.
            face_confidence = face[2]        
            # Check if the face detection confidence score is greater than the thresold.
            if face_confidence > 0.5:#min_confidence=0.5
                # Retrieve the bounding box of the face.
                bbox = face[3:]
                # Retrieve the bounding box coordinates of the face and scale them according to the original size of the image.
                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)
                # Draw a bounding box around a face on the copy of the image using the retrieved coordinates.
                cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//200)
                    
                # Draw a empty rectangle near the bounding box of the face.
                # We are doing it to change the background of the confidence score to make it easily visible.
                cv2.rectangle(frame, pt1=(x1, y1-image_width//20), pt2=(x1+image_width//16, y1),
                                  color=(0, 255, 0), thickness=1)

                # Write the confidence score of the face near the bounding box and on the filled rectangle. 
                cv2.putText(frame, text=str(round(face_confidence, 1)), org=(x1, y1-25), 
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.40,
                                color=(0,0,255), thickness=2)
            
          
        cv2.imshow('Detected faces', frame) 
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video_cascade.release()
    cv2.destroyAllWindows()        

# Load a model stored in TensorFlow framework's format using the architecture and the layers weights file stored in the disk       
opencv_dnn_model = cv2.dnn.readNetFromTensorflow(model=r"C:\Users\user\Desktop\AAmer works\opencv_face_detector_uint8.pb", config=r"C:\Users\user\Desktop\AAmer works\opencv_face_detector.pbtxt")
video_cascade = cv2.VideoCapture(0)

face_detection(video_cascade,opencv_dnn_model)
