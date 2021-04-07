from __future__ import division


from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
import os
import sys
import time
import datetime
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

#model, 경로, iou_threshold, confidence threshold, non max threshold, image 사이즈, batch 사이즈를 입력받아 model의 성능을 평가한다.
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    
    model.eval() #will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

    
    dataset = ListDataset(path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS) #ListDataset 객체 생성
    dataloader = torch.utils.data.DataLoader(
        dataset,                            
        batch_size=batch_size,          
        shuffle=False,                      #epoch 마다 data를 섞는다.  Use shuffle (the data reshuffled at every epoch)
        num_workers=1,                      # 프로세스 worker 설정
        collate_fn=dataset.collate_fn       # 리스트들을 합치는 collate_fn 할당
    )
 
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor #Tensor를 할당

    labels = [] # label 값들을 저장하는 리스트 생성
    sample_metrics = [] #  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")): # for 문을 돌려 model 평가하기 시작한다
        
        if targets is None:
            continue
            
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])# target 안의 값들을 재 설정해준다.
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)# image를 초기화 한다.

        
        with torch.no_grad():
            outputs = to_cpu(model(imgs))
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres) #non-max suppression 함수를 실행한다. 

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres) # 출력값 ,예측값, iou_threshold 값을 get_batch_statistics함수에 넣어서 리턴값을 sample_metrics 리스트에 저장
    
    if len(sample_metrics) == 0:  #평가가 안되면 None을 리턴한다.
        return None
    
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]# 실제값, 예측값의 점수, 예측값의 라벨에 값을 할당한다.
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels) #precision,recall,average precision 등을 계산한다.

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() #ArgumentParser 객체를 생성하여 option값들을 받는다.
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CUDA 가 사용가능한지 보고 불가능하면 CPU를 할당한다.

    data_config = parse_data_config(opt.data_config) #config 파일을 파싱한다.
    valid_path = data_config["valid"] # #config 파일의 "names"부분을 값으로 할당한다.
    class_names = load_classes(data_config["names"]) #config 파일의 "names"부분을 값으로 할당한다.

    model = Darknet(opt.model_def).to(device) # Darknet 객체 생성
    if opt.weights_path.endswith(".weights"): # 만약 weights_path가 .weigths 확장자로 끝나면 model객체의 load_darknet_weights 매쏘드 호출
        model.load_darknet_weights(opt.weights_path)
    else: # 아닐경우 model객체의 load_state_dic 매쏘드 호출
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    # evaluate 함수를 호출하여 precision, recall, AP, ap_class 값을 반환한다.
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class): #class 별 average precision 호출 
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}") #AP의 평균값인 mAP값을 계산하여 출력한다.
