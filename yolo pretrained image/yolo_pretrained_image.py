# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:54:42 2022

@author: DilaraSevimPolat
"""
import cv2
import numpy as np

img=cv2.imread("C:\\\YOLO\\yolo pretrained image\\images\\people.jpg")#BGR hali


img_widht=img.shape[1]#enini aldım
img_height=img.shape[0]#boyunu aldım

img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)#yolo algoritmasını kullanmak icin resimi blob formata ceviriyorum
#resmim, scalefactor yolo yazarları bunun dogru degeri 1/255 olarak bulmus, egitimim 413,413 resimler yani kullndiğim model, resmimi BGR dan RGB'ye ceviriyorum.
#,crop=false resmim kırpılmasın

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
          "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
          "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
          "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
          "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
          "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
          "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
          "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
          "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
          "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
#80 tane label var


colors=["0,255,255","0,0,255","255,0,0","255,255,0","255,255,255","0,255,0"]
colors=[np.array(color.split(",")).astype("int")for color in colors]
colors=np.array(colors)
colors=np.tile(colors,(18,1))#5 kere 1 kez yan yana ekleme tail cogaltma isleminde kullanıllıyor
 #        #%% 3.Bölüm  ile kodu bölümlere ayırabiliyoruz
model=cv2.dnn.readNetFromDarknet("C:/YOLO/pretrained_model/yolov3.cfg","C:/YOLO/pretrained_model/yolov3.weights")

layers=model.getLayerNames()#layers degisşkenine modeldeki layers atadım
output_layer=[layers[layer - 1] for layer in model.getUnconnectedOutLayers()]#Layers hespsine ihtiyacım yok sadece cıktıları alacagım
#model.getUnconnectedOutLayers() ->  array([200, 227, 254]) bu degrelerin 1 eksiginde yolo cıktım katmanım var

model.setInput(img_blob)

detection_layers=model.forward(output_layer)#cıktı katmanımdaki (output_layer) bir takım degerlere erişiyorum



for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores=object_detection[5:]
        predicted_id=np.argmax(scores)#maksimum degerdeki ndeksi veriyor
        confidence=scores[predicted_id]
        
        if confidence > 0.99:
            label=labels[predicted_id]
            bounding_box=object_detection[0:4]*np.array([img_widht,img_height,img_widht,img_height])
            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")#float degerlerle calısamıyorum int olması gerekiyor
            
            start_x=int(box_center_x-(box_width/2))
            start_y=int(box_center_y-(box_height/2))
            
            end_x=start_x+box_width
            end_y=start_y+box_height
            
            box_color=colors[predicted_id]
            box_color=[int(each) for each in box_color]
            
            
            
            label="{}:{:.2f}%".format(label, confidence*100)
            
            print("predicted object {}".format(label))
            
            cv2.rectangle(img, (start_x, start_y),(end_x,end_y),box_color,1)
            cv2.putText(img, label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)


cv2.imshow("Detection Window",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


























