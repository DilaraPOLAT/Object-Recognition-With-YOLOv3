# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:54:42 2022

@author: DilaraSevimPolat
"""
import cv2
import numpy as np

cap=cv2.VideoCapture(0)#BGR hali


while True:
    ret, frame=cap.read()
    if ret==0:
        break
    
    frame_widht=frame.shape[1]#enini aldım
    frame_height=frame.shape[0]#boyunu aldım

    frame_blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)#yolo algoritmasını kullanmak icin resimi blob formata ceviriyorum
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

    model.setInput(frame_blob)

    detection_layers=model.forward(output_layer)#cıktı katmanımdaki (output_layer) bir takım degerlere erişiyorum

############################ Non Maximim Suppression  Oeration-1 ###########################☺
    ids_list=[]
    bocces_list=[]
    confidences_list=[]




############################ END OF Non Maximim Suppression  Oeration-1 ###########################☺




    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores=object_detection[5:]
            predicted_id=np.argmax(scores)#maksimum degerdeki ndeksi veriyor
            confidence=scores[predicted_id]
            
            if confidence > 0.99:
                label=labels[predicted_id]
                bounding_box=object_detection[0:4]*np.array([frame_widht,frame_height,frame_widht,frame_height])
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")#float degerlerle calısamıyorum int olması gerekiyor
                
                start_x=int(box_center_x-(box_width/2))
                start_y=int(box_center_y-(box_height/2))
                ############################ Non Maximim Suppression  Oeration-2 ###########################☺
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                bocces_list.append([start_x,start_y,int(box_width),int(box_height)])
                
                ############################ END OF Non Maximim Suppression  Oeration-2 ###########################☺
                
    ############################ Non Maximim Suppression  Oeration-3 ###########################☺
    max_ids=cv2.dnn.NMSBoxes(bocces_list, confidences_list, 0.5, 0.4)#♥makimium boudibox bir array içinde veriyor. önerilen thrashold degeri 0.5,0.4
    
    
    for max_id in max_ids:
        max_class_id=max_id
        box=bocces_list[max_class_id]
    
        start_x=box[0]
        start_y=box[1]
        box_width=box[2]
        box_height=box[3]
        
        predicted_id=ids_list[max_class_id]
        label=labels[predicted_id]
        confidence=confidences_list[max_class_id]
        
    
    ############################ END OF Non Maximim Suppression  Oeration-3 ###########################☺
                
        end_x=start_x+box_width
        end_y=start_y+box_height
                
        box_color=colors[predicted_id]
        box_color=[int(each) for each in box_color]
                
                
                
        label="{}:{:.2f}%".format(label, confidence*100)
                
        print("predicted object {}".format(label))
                
        cv2.rectangle(frame, (start_x, start_y),(end_x,end_y),box_color,1)
        cv2.putText(frame, label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
    
    
    cv2.imshow("Detection Window",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cv2.destroyAllWindows()
cap.release()


























