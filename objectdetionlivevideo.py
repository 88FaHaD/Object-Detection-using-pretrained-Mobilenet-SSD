import cv2 as cv

threshold = 0.5
classnames = []

with open('coco.names', 'r') as file:
    for line in file:
        classnames.append(line.strip())

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'
net = cv.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgrsize = cv.resize(frame, (480, 320))

    classIds, confs, bbox = net.detect(imgrsize, threshold)

   
    for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
     cv.rectangle(imgrsize, box, (0, 255, 0), 2)
     object_name = classnames[classId - 1] if classId > 0 else "Unknown"  # Adjusting for 1-based indexing
     confidence_str = f"Confidence: {conf:.2f}"
     cv.putText(imgrsize, object_name, (box[0] + 5, box[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     cv.putText(imgrsize, confidence_str, (box[0] + 5, box[1] + 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('window', imgrsize)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
