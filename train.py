from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from utils.loss import compute_loss
from test import evaluate

from terminaltables import AsciiTable

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser() #ArgumentParser 객체 생성
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs") #epochs 지정 defaulst 값 300
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") #model 지정, 구조 yolov3.configure 파일로부터 받아오기
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file") #coco.data 파일로 부터 class 갯수 train_path, valid_path 지정하기
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")#이미 학습된 모델의 parameter(weights,bias)를 사용한다
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #cpu 갯수를 지정한다 
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension") #img 사이즈 parsing default size=416
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights") #check point interval parsing, default interval=1
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")#ev point interval parsing, default interval=1
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")#iteration 마다 image size를 resize해주는 multiscale_training 파싱
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored") #log file들이 저장될 곳을 parsing
    opt = parser.parse_args() #parse_arg 객체 생성
    print(opt)

    logger = Logger(opt.logdir) #logger 객체 생성

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# torch.tensor 객체가 할당될 곳을 지정, 하나의 gpu 사용
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    os.makedirs("output", exist_ok=True)#ouput폴더 생성 exist_ok를 지정하면 폴더가 존재하지 않으면 생성하고, 존재하는 경우에는 아무것도 하지 않는다
    os.makedirs("checkpoints", exist_ok=True)#checkpoints폴더 생성 exist_ok를 지정하면 폴더가 존재하지 않으면 생성하고, 존재하는 경우에는 아무것도 하지 않는다

    # Get data configuration
    data_config = parse_data_config(opt.data_config) #parse_data_config 함수를 사용하여 path를 받아온후 dictionary 자료형으로 변형 
    train_path = data_config["train"] # data_config dictionary의 key="train"인 value를 train_path에 str 형태로 저장 
    valid_path = data_config["valid"] # data_config dictionary의 key="valid"인 value를 valid_path에 str 형태로 저장 
    class_names = load_classes(data_config["names"])# data_config dictionary의 key="names"인 value를 load_classes 함수에 넣어 class_names 변수에 저장-> class 종류들일 리스트 형태로 저장된다

    # Initiate model
    model = Darknet(opt.model_def).to(device)  # Darknet 객체 생성 후 어느 device에 올릴지 지정, Darknet class 는 pytorch nn.module 상속, otp.model_def 값(config 파일 경로)를 전달받는
    model.apply(weights_init_normal) # apply 함수에 weights_init_normal 함수를 인자로 입력받아 다시 wieghts_init_normal 함수를 호출하고  각 layer들(Conv,BatchNorm2d)의 parameter 초기화를 해준다. 

    # If specified we start from checkpoint
    if opt.pretrained_weights: #pretrained 된 모델이 있는 경우 
        if opt.pretrained_weights.endswith(".pth"): #pth 확장자 파일이 있는 경우
            model.load_state_dict(torch.load(opt.pretrained_weights))#load_state_dict를 호출하여 pytorch의 data를 python에서 사용할 수 있도록 deserialized한다.
        else:
            model.load_darknet_weights(opt.pretrained_weights)#load_darknet_weights를 호출하여 사용한다.

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS) #ListDataset 객체 dataset을 생성
    # torch.utils.data.DataLoader 객체 dataloader 생성 -> dataset을 불러온다
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= model.hyperparams['batch'] // model.hyperparams['subdivisions'],#how many samples per batch to load
        shuffle=True,
        num_workers=opt.n_cpu, #how many subprocesses to use for data loading
        pin_memory=True,
        collate_fn=dataset.collate_fn,#merges a list of samples to form a mini-batch of Tensor
    )

    if (model.hyperparams['optimizer'] in [None, "adam"]): #만약 yolov3.cfg 안에 optimizer의 값이 None 또는 adam이면 Adam optimizer 사용
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'], 
            weight_decay=model.hyperparams['decay'],
            )
    elif (model.hyperparams['optimizer'] == "sgd"):#만약 yolov3.cfg 안에 optimizer의 값이 sgd이면 Adam optimizer 사용
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).") # optimizer에 다른 값이 들어가면 에러 메세지

    for epoch in range(opt.epochs): #epoch 설정
        print("\n---- Training Model ----")
        model.train() # train 시작
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")): # 미니배치만큼 돌린다
            batches_done = len(dataloader) * epoch + batch_i #len dataloader=minibatch 갯수 , batches_done은 epch의 마지막까지 학습에 이용된 minibatch의 갯수-1이 저장된다.

            imgs = imgs.to(device, non_blocking=True) #cpu에 있는image를 gpu로 올린다.
            targets = targets.to(device) #cpu에 있는 label을 gpu로 올린다.

            outputs = model(imgs) # 

            loss, loss_components = compute_loss(outputs, targets, model) # 예측값과 실제값, 모델을 넣어서 loss 값을 계산한다.

            loss.backward() # loss 값을 backpropagation을 

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0: # batches_done이 subdivision의 값으로 나누어 떨어지면 아래 조건문 실행-> gpu에 minibatch가 다 안올라가는 경우
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']# lr에 learning_rate값 저장
                if batches_done < model.hyperparams['burn_in']: # default burn_in=1000, 만약 bathes_done이 burn in 보다 작을 경우 learning rate를 점점 작게 하여 학습속도 감소시킨다.
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']: #만약 bathes_done이 threshold보다 크면 learning rate에 value를 계속 곱한다.
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done) #learning rate를 log 메세지로  표현
                # Set learning rate
                for g in optimizer.param_groups: #optimizer 안의 parm_gropus에 learning rate 저장
                        g['lr'] = lr

                # Run optimizer
                optimizer.step() #매개변수 갱신
                # Reset gradients
                optimizer.zero_grad() #역전파 단계를 실행하기 전에 변화도를 0으로 만든다.

            # ----------------
            #   Log progress #log에 loss_components의 요소들을 나타낸다.
             # ----------------
            log_str = "" 
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table

            if opt.verbose: print(log_str)

            # Tensorboard logging #Tensorboard에 loss_components요소들을  나타낸다.
            tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])), 
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0: #epoch이 parsing 한 evaluation_interval 값의 배수일 때 아래 조건문 실행  
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set # validation set에 model, valid dataset 경로, iou_threshold 값, confidence_threshold, non-max threshold, img,size,batch_size 을 입력하여
            #test.py 내부의 evaluate 함수로 validation set을 갖고 모델을 평가한다. 
            metrics_output = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.1,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],
            )
            
            if metrics_output is not None: 
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                if opt.verbose:
                    # class AP와 mAP를 출력한다.
                    ap_table = [["Index", "Class name", "AP"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- mAP {AP.mean()}")                
            else:
                print( "---- mAP not measured (no detections found by model)")

        if epoch % opt.checkpoint_interval == 0: # chckpint를 저장해 다음 학습시에 이용할 수 있는 파일을 저장한다.
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
