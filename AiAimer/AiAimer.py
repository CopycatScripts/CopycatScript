import cv2
from ControllerMapping import Buttons
from Macro import Macro
from ScriptCore import Template, PrintColors
import random
import numpy as np

class Script(Template):
    def __init__(self, controller, report):
        super().__init__(controller, report)

        # load weights model for image detection
        self.weights = 'scripts/AiAimer/weights/GENERIC.weights'
        self.config = 'scripts/AiAimer/weights/GENERIC.cfg'
        self.classes = ['target']
        self.trackingX = 0
        self.trackingY = 0
    
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.classes = []

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

        # Try enable fp16 for faster detection
        try:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Get the output layer names
        self.ln = self.net.getLayerNames()

        # Backwards compatibility between old and new cuda version
        try:
            self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.print_log('Warzone model loaded!', PrintColors.COLOR_GREEN)

        self.rapid_fire = Macro(controller, [
            [self.right_trigger_float, 0],
            ["wait_random", 50, 150],
            [self.right_trigger_float, 1],
            ["wait_random", 50, 150],
            [self.right_trigger_float, 0],
        ])
         self.auto_melee = Macro(controller, [
            [self.release_button, Buttons.BTN_RIGHT_THUMB],
            ["wait_random", 50, 150],
            [self.press_button, Buttons.BTN_RIGHT_THUMB],
            ["wait_random", 50, 150],
            [self.release_button, Buttons.BTN_RIGHT_THUMB]
        ])

    def getDistance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def run(self, frame):
        moveX = self.get_actual_right_stick_x()
        moveY = self.get_actual_right_stick_y()

        detectionSize = int(self.get_setting('performance'))

        height, width, channels = frame.shape

        if (height == 0):
            return frame
            
        if (self.get_setting('hair_trigger')):
            # Hair trigger left
            if (self.get_actual_left_trigger() > 0.01):
                self.left_trigger_float(1)

            # Hair trigger right
            if (self.get_actual_right_trigger() > 0.01 and self.rapid_fire.isStopped()):
                self.right_trigger_float(1)

       if (self.get_setting('auto_melee')):
            # Auto Melee 
            if (self.is_actual_button_pressed(Buttons.BTN_RIGHT_THUMB)):
                self.auto_melee.run()

       if (self.get_setting('rapid_fire')):
            # Rapid Fire
            if (self.get_actual_right_trigger() > 0.01):
                self.rapid_fire.run()

        # Generate some variables for the frame
        adsTopX = round(int(width/2) - int(detectionSize/2))
        adsTopY = round(int(height/2) - int(detectionSize/2))
        adsBottomX = round(int(width/2) + int(detectionSize/2))
        adsBottomY = round(int(height/2) + int(detectionSize/2))
        frameCenterX = round(width / 2)
        frameCenterY = round(height / 2)

        # Crop down frame
        scanFrame = frame.copy()
        scanFrame = scanFrame[ adsTopY:adsBottomY , adsTopX:adsBottomX ]

        # Detecting objects
        blob = cv2.dnn.blobFromImage(scanFrame, 1 / 255.0, (detectionSize, detectionSize), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        # Convert layerOutputs into a single numpy array
        layerOutputs = np.vstack(layerOutputs)

        # Extract class IDs, confidence scores, and bounding box coordinates
        scores = layerOutputs[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        # Filter out bounding boxes with confidence less than confCap
        high_confidence_mask = confidences > float(self.get_setting('detection_accuracy'))
        filtered_layerOutputs = layerOutputs[high_confidence_mask]

        # Calculate the bounding box coordinates for the filtered detections
        width = scanFrame.shape[1]
        height = scanFrame.shape[0]
        center_x = filtered_layerOutputs[:, 0] * width
        center_y = filtered_layerOutputs[:, 1] * height
        w = filtered_layerOutputs[:, 2] * width
        h = filtered_layerOutputs[:, 3] * height

        x = (center_x - w / 2).astype(int)
        y = (center_y - h / 2).astype(int)
        x2 = (x + w).astype(int)
        y2 = (y + h).astype(int)

        # Create a list of filtered bounding boxes with confidence and class IDs
        filtered_boxes = list(zip(x, y, x2, y2, confidences[high_confidence_mask], class_ids[high_confidence_mask]))
        # turn on to enable non-max suppression (or leave off for fps boosts)
        # filtered_boxes = self.non_max_suppression(filtered_boxes, 0.5)
        
        closest_distance = float('inf')
        closest_box = None

        targetColour = (0, 0, 255)
        targetSize = 2

        # Loop through filtered boxes and find the closest one to the center of the frame
        for box in filtered_boxes:
            x, y, x2, y2, score, classID = box
            
            width = x2 - x
            height = y2 - y
            
            box_center_x = round(adsTopX+x+(width/2))
            box_center_y = round(adsTopY+y+(height/2))

            # Get distance to center
            distance = self.getDistance(frameCenterX, frameCenterY, box_center_x, box_center_y)

            if distance < closest_distance:
                closest_distance = distance
                closest_box = box
            else:
                continue

        if (closest_box is not None):
            x, y, x2, y2, score, classID = closest_box

            width = x2 - x
            height = y2 - y
            
            box_center_x = round(adsTopX+x+(width/2))
            box_center_y = round(adsTopY+y+(height/2))
            box_center_y = (adsTopY+y) + round(height * float(self.get_setting('target_offset')))

            # Draw tracker dot
            cv2.circle(frame, (box_center_x, box_center_y), 1, targetColour, 5)

            # Draw line from box to center
            cv2.line(frame, (box_center_x, box_center_y), (frameCenterX, frameCenterY), targetColour, targetSize)
            
            # Draw rectangle around enemy
            cv2.rectangle(frame, (adsTopX+x, adsTopY+y), (adsTopX+x2, adsTopY+y2), targetColour, targetSize)

            # Draw sticky bubble around enemy
            stickBubbleSizeM = round((width+height) / 2)
            cv2.circle(frame, (box_center_x, box_center_y), stickBubbleSizeM, (255, 0, 0), 1)

            if (self.get_actual_left_trigger() > 0):
                # Calculate the distance from the center of the frame to the center of the enemy box
                distance_x = frameCenterX - box_center_x
                distance_y = frameCenterY - box_center_y

                distance = self.getDistance(frameCenterX, frameCenterY, box_center_x, box_center_y)

                # Calculate the distance from the center of the frame to the center of the enemy box
                self.trackingX = round(self.map_range(distance_x, -int(detectionSize/2), int(detectionSize/2), -1, 1), 2)
                self.trackingY = round(self.map_range(distance_y, -int(detectionSize/2), int(detectionSize/2), -1, 1), 2)

                if (distance < stickBubbleSizeM):
                    self.trackingX = np.clip(self.trackingX * (float(self.get_setting('tracking_speed')) + float(self.get_setting('sticky_bubble_speed'))), -1, 1)
                    self.trackingY = np.clip(self.trackingY * (float(self.get_setting('tracking_speed')) + float(self.get_setting('sticky_bubble_speed'))), -1, 1)
                else:
                    self.trackingX = np.clip(self.trackingX * float(self.get_setting('tracking_speed')), -1, 1)
                    self.trackingY = np.clip(self.trackingY * float(self.get_setting('tracking_speed')), -1, 1)

                # set deadzone
                deadzoneX = float(self.get_setting('deadzone'))
                deadzoneY = float(self.get_setting('deadzone'))

                if self.trackingX < 0:
                    deadzoneX = -deadzoneX

                if self.trackingY < 0:
                    deadzoneY = -deadzoneY

                self.trackingX = np.clip(self.trackingX + deadzoneX, -1, 1)
                self.trackingY = np.clip(self.trackingY + deadzoneY, -1, 1)
            else:
                self.trackingX = 0
                self.trackingY = 0

        self.right_joystick_float(np.clip(moveX + -self.trackingX, -1, 1), np.clip(moveY + self.trackingY, -1, 1))
        
        # Handle decay
        if (self.trackingX > 0):
            self.trackingX = np.clip(self.trackingX - 0.01, 0, 1)

        if (self.trackingX < 0):
            self.trackingX = np.clip(self.trackingX + 0.01, -1, 0)

        if (self.trackingY > 0):
            self.trackingY = np.clip(self.trackingY - 0.01, 0, 1)

        if (self.trackingY < 0):
            self.trackingY = np.clip(self.trackingY + 0.01, -1, 0)
        
        self.auto_melee.cycle()

        self.rapid_fire.cycle()


        return frame

        return frame

