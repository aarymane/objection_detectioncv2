import cv2
#from gui_buttons import Buttons
#
#button = Buttons()
#button.add_button("person", 5, 10)
#button.add_button("cell phone", 5, 50)
#button.add_button("keyboard", 5, 90)
#button.add_button("remote", 5, 130)
#button.add_button("scissors", 5, 170)

#Colors = button.colors

net = cv2.dnn.readNet("C:\\Users\\91932\\PycharmProjects\\pythonProject\\dnn_model\\yolov4-tiny.weights",
                      "C:\\Users\\91932\\PycharmProjects\\pythonProject\\dnn_model\\yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# load class list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


#def click_button(event, x, y, flags, params):
    #global button_person
    #if event == cv2.EVENT_LBUTTONDOWN:
        #button.button_click(x, y)



# Create window
cv2.namedWindow("Frame")
#cv2.setMouseCallback("Frame", click_button)

while True:

    # Get active buttons list
    #active_buttons = button.active_buttons_list()
    # print("Active buttons", active_buttons)

    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(
        frame, confThreshold=0.3, nmsThreshold=0.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        cv2.putText(frame, class_name, (x, y - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    # Display buttons
    #button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
