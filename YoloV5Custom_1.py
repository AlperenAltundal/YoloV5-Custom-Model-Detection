#Ahmet Alperen Altundal_1 13/12/2022

import torch
import numpy as np
import cv2
from time import time


class CustomDetector:
    

    def __init__(self, capture_index, model_name):

        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
      
        return cv2.VideoCapture("C:/Users/pc/vs_code/Z11_yolo_wx/f1_2_Trim.mp4 ")


    def load_model(self, model_name):

        if model_name:
            model = torch.hub.load('C:/Users/pc/yolov5-master', 'custom', source='local', path=model_name, force_reload=True)
        else:
            model = torch.hub.load("C:/Users/pc/yolov5-master/data", 'custom', path='best2.pt', source='local')

        return model

    def score_frame(self, frame):
        
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                

        return frame

    def __call__(self):
        
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
              
            ret, frame = cap.read()
            assert ret
            
            frame = cv2.resize(frame, (1280,720))
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)         
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
      
        cap.release()
        cv2.destroyAllWindows()
        
alperen='yolov5s.pt'
alperen="C:/Users/pc/vs_code/Z11_yolo_wx/best2.pt"

detector = CustomDetector(capture_index=0, model_name=alperen)
detector()


