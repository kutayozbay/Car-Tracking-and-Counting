import os
import ultralytics
from ultralytics import YOLO
import supervision as sv
import numpy as np
from tqdm.notebook import tqdm
import csv

print(ultralytics.checks())


HOME = os.getcwd()
print(HOME)


SOURCE_VIDEO_PATH = f"{HOME}/video.mp4"


MODEL = "yolov8x.pt"

model = YOLO(MODEL)
model.fuse()



# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
selected_classes = [2, 3, 5, 7]

if SOURCE_VIDEO_PATH == f"{HOME}/video1.MOV":   
    # settings
    LINE_START = sv.Point(320, 1300)
    LINE_END = sv.Point(3040, 1300)

elif SOURCE_VIDEO_PATH == f"{HOME}/video2.MOV":   
    # settings
    LINE_START = sv.Point(320, 650)
    LINE_END = sv.Point(1520, 650)

elif SOURCE_VIDEO_PATH == f"{HOME}/video.mp4": 
    LINE_START = sv.Point(120, 450)
    LINE_END = sv.Point(1020, 450)


TARGET_VIDEO_PATH = f"{HOME}/video_target_2.MOV"


sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)



# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

#print(video_info)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)


# create LineZoneAnnotator instance
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# create a list to store the data
car_data_list = []
old_out = 0
old_in = 0
car_id_list = []
old_detections_xyxy = []
old_detections_ids = []


# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    global old_in
    global old_out
    global car_id_list
    global old_detections_xyxy
    global old_detections_ids
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    #print(trace_annotator.trace.xy)
    #print(trace_annotator.trace.tracker_id)

    annotated_frame=box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)    
    
    
    # update line counter
    crossed_in, crossed_out = line_zone.trigger(detections)

    if old_out != line_zone.out_count or old_in != line_zone.in_count: 
        if old_out != line_zone.out_count:
            for i in range(len(crossed_out)):
                if crossed_out[i] == True:
                    car_id = detections.tracker_id[i]  
                    break    
            print("UP")
            car_movement = "UP"
            old_out = line_zone.out_count
            car_timestamp = index / video_info.fps
            car_data_list.append([car_id, car_timestamp, car_movement])
        if old_in != line_zone.in_count:
            for i in range(len(crossed_in)):
                if crossed_in[i] == True:
                    car_id = detections.tracker_id[i]
                    break            
            print("DOWN")
            car_movement = "DOWN"
            old_in = line_zone.in_count
            car_timestamp = index / video_info.fps
            car_data_list.append([car_id, car_timestamp, car_movement])

    if len(old_detections_xyxy) == 0:
        old_detections_xyxy = detections.xyxy
        old_detections_ids = detections.tracker_id
    else:
        i = 0
        while i < len(old_detections_ids):
            if old_detections_ids[i] not in detections.tracker_id:
                new_array_id = np.delete(old_detections_ids, i)
                old_detections_ids = new_array_id
                old_detections_xyxy = np.array(old_detections_xyxy)
                rows_to_remove = [i]
                new_arr = old_detections_xyxy[np.logical_not(np.isin(np.arange(old_detections_xyxy.shape[0]), rows_to_remove))]
                old_detections_xyxy = new_arr
                i -= 1
                if len(old_detections_ids) == 0:
                    break
                continue
            if i < 0:
                i == 0
            else:
                i += 1
        i = 0
        while i < len(detections.tracker_id):
            if detections.tracker_id[i] not in old_detections_ids:
                new_array_id = np.delete(detections.tracker_id, i)
                detections.tracker_id = new_array_id
                rows_to_remove = [i]
                new_arr = detections.xyxy[np.logical_not(np.isin(np.arange(detections.xyxy.shape[0]), rows_to_remove))]
                detections.xyxy = new_arr
                i -= 1
                if len(old_detections_ids) == 0:
                    break
                continue
            if i < 0:
                i == 0
            else:
                i += 1
        for i in range(len(old_detections_ids)):
            w_old, h_old = old_detections_xyxy[i][2] - old_detections_xyxy[i][0], old_detections_xyxy[i][3] - old_detections_xyxy[i][1]
            cx_old, cy_old = old_detections_xyxy[i][0] + w_old // 2, old_detections_xyxy[i][1] + h_old //2
            w, h = detections.xyxy[i][2] - detections.xyxy[i][0], detections.xyxy[i][3] - detections.xyxy[i][1]
            cx, cy = detections.xyxy[i][0] + w // 2, detections.xyxy[i][1] + h //2
            car_timestamp = index / video_info.fps         
            """
            print("---------------------------")
            print("cx_old:", cx_old)
            print("cy_old:", cy_old)
            print("cx:", cx)
            print("cy:", cy)
            print("car_id:", detections.tracker_id[i])
            print("time_stamp:",car_timestamp)
            print("---------------------------")
            """
            if cy_old < cy:
                if cx > cx_old + 20:
                    car_movement = "RIGHT"
                    print("RIGHT")
                    car_timestamp = index / video_info.fps
                    car_id = detections.tracker_id[i]
                    car_data_list.append([car_id, car_timestamp, car_movement])
                if cx < cx_old - 20:
                    car_movement = "LEFT"
                    print("LEFT")
                    car_timestamp = index / video_info.fps
                    car_id = detections.tracker_id[i]
                    car_data_list.append([car_id, car_timestamp, car_movement])
            if cy_old > cy:
                if cx > cx_old + 20:
                    car_movement = "LEFT"
                    print("LEFT")
                    car_timestamp = index / video_info.fps
                    car_id = detections.tracker_id[i]
                    car_data_list.append([car_id, car_timestamp, car_movement])
                if cx < cx_old - 20:
                    car_movement = "RIGHT"
                    print("RIGHT")
                    car_timestamp = index / video_info.fps
                    car_id = detections.tracker_id[i]
                    car_data_list.append([car_id, car_timestamp, car_movement])
            
        old_detections_ids = detections.tracker_id
        old_detections_xyxy = detections.xyxy
    # return frame with box and line annotated result
    return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# process the whole video
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)

csv_file_name ='car_data_video_2.csv'
with open(csv_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(car_data_list)