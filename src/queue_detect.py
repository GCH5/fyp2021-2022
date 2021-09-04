import argparse

from pathlib import Path
import numpy as np
import cv2
import datetime
from tqdm import tqdm
import torch
from shapely.geometry import Polygon
from shapely.geometry import Point

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from my_utils.encoder import create_box_encoder
from models.experimental import attempt_load
from my_utils.my_dataset import LoadImages
from my_utils import utils
from deep_sort import detection



def run(
    weights='yolov5l.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    line=[0, 300, 1000, 200], # boundary crossing line
    queue_polygon=[526,215,1106,929],   # x y x y x y x y x y
    debug_frames=0, # debug mode
    half=False,  # use FP16 half-precision inference
    save_img=False,
    save_video=False,
):

    line = ((line[0], line[1]),(line[2], line[3]))
    vertices = []
    for x, y in zip(*[iter(queue_polygon)]*2):   # loop 2 coords at a time
        vertices.append((x,y))
    queue_polygon = Polygon(vertices)
    device = utils.select_device(device)
    use_gpu = device == torch.device('cuda:0')
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    model.conf = conf_thres
    model.iou = iou_thres
    if half:
        model.half()
    
    encoder = create_box_encoder('mars-small128.pb', batch_size=32)
    max_cosine_distance = 0.2
    nn_budget = None
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=640, stride=stride)

    out_counter = 0 # below -> above
    in_counter = 0 # above -> below
    memory = {}

    dir_path = Path(output_dir)
    file_path = Path('output.txt')
    dir_path.mkdir(exist_ok=True)
    p = Path(output_dir) / file_path
    with p.open('a') as f:
        if save_video:
            for _, img, im0s, _, frame_idx in dataset:
                height, width, _ = im0s.shape
                break
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(str(dir_path / Path("out.mp4")), fourcc, 60, size)
        for _, img, im0s, _, frame_idx in tqdm(dataset):
            if debug_frames > 0:
                if frame_idx > debug_frames:
                    break
            
            indexIDs = []
            boxes = []
            previous = memory.copy() # last frame we have these ppl boxes
            memory = {}

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            bgr_image = im0s

            
            results = model(img)[0]
            results = utils.non_max_suppression(results)
            if(results[0].shape[0]==0):
                continue

            det = results[0]
            det[:, :4] = utils.scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            person_ind = [i for i, cls in enumerate(det[:, -1]) if int(cls) == 0]   
            xyxy = det[person_ind, :-2]  # find person only
            xywh_boxes = utils.xyxy2xywh(xyxy)
            tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
            confidence = det[:, -2]
            if use_gpu:
                tlwh_boxes = tlwh_boxes.cpu()
            features = encoder(bgr_image, tlwh_boxes)
            
            detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)


            ppl_count = 0   
            queue_list = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                indexIDs.append(track.track_id) # # this frame we have these ppl idxs
                memory[track.track_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]  # this frame we have these ppl boxes
                ppl_count += 1
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                if queue_polygon.intersects(Point(center_x, center_y)):
                    queue_list.append(track.track_id)
                    if save_img or save_video:
                        cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                        pts=np.array(vertices,np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(bgr_image,[pts],True,(255,0,0), 2) 
                        cv2.putText(bgr_image, "ID: " + str(track.track_id), (int(center_x), int(center_y)), 0,
                                            1e-3 * bgr_image.shape[0], (0, 255, 0), 1)


            if ppl_count > 0:
                for i, box in enumerate(boxes):
                    if indexIDs[i] in previous:
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        p0 = (center_x, center_y)
                        previous_box = previous[indexIDs[i]]
                        center_x2 = int((previous_box[0] + previous_box[2]) / 2)
                        center_y2 = int((previous_box[1] + previous_box[3]) / 2)
                        p1 = (center_x2, center_y2)

                        # cv2.line(bgr_image, p0, p1, (0, 255, 0), 3)

                        if utils.intersect(p0, p1, line[0], line[1]):
                            if utils.below_line(line, p0): 
                                out_counter += 1
                            else: 
                                in_counter += 1

                f.write(str(frame_idx))
                f.write(",")
                f.write(str(datetime.timedelta(seconds=frame_idx//25)))
                f.write(",")
                f.write(str(ppl_count))
                f.write(",")
                f.write(str(in_counter + out_counter)) # accumulated flux
                f.write(",")
                for trackid in indexIDs:
                    f.write(" ")
                    f.write(str(trackid))
                f.write("\n")
                if save_img or save_video:                        
                    image_path = dir_path / Path('{:06}.jpg'.format(frame_idx) + ".jpg")
                    cv2.putText(bgr_image, "Queue Lenth: {}".format(str(len(queue_list))), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(bgr_image, "Queue: {}".format(str(queue_list)[1:-1]), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if save_video:
                      video_writer.write(bgr_image)
                if save_img:
                    cv2.imwrite(str(image_path), bgr_image)
        if save_video:
            video_writer.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line', nargs='+', type=int, default=[0, 300, 1000, 200], help='boundary crossing line')
    parser.add_argument('--queue-polygon', nargs='+', type=int, default=[526,215,1106,929], help='queue area')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    parser.add_argument('--save-img', action='store_true', help='save detection output as images')
    parser.add_argument('--save-video', action='store_true', help='save detection output as an video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)