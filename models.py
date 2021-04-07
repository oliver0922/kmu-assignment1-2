from __future__ import division
from itertools import chain

# Import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Import utils
from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs): #moduele_defs =[{"type":"net","batches":16,"subdivision":1,...},{"type":"convolutional","batch_normalize":1,"filters":32",...}]
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0) #moudel 리스트의 첫번째 요소를 pop하여 hyperparams(dictionary 자료형)에 저장한다. 
    hyperparams.update({ #hyperparmas를 인덱싱하여 값을 대입한다.
        'batch': int(hyperparams['batch']), #batch 사이즈
        'subdivisions': int(hyperparams['subdivisions']), # minibatch를 더 잘게 쪼갠 단위
        'width': int(hyperparams['width']),# 이미지 너비 여기서는 416
        'height': int(hyperparams['height']),#이미지 높이 여기서는 416
        'channels': int(hyperparams['channels']), # 사진의 채널 갯수 여기서는 rgb이므로 3
        'optimizer': hyperparams.get('optimizer'), # optimizer 
        'momentum': float(hyperparams['momentum']), # momentum 학습속도를 빠르게, 부드럽게 해주는 요소
        'decay': float(hyperparams['decay']), # learning rate 감소시키는 비율
        'learning_rate': float(hyperparams['learning_rate']), # learning rate =>학습율 보통 올라갈수록 학습속도가 빨라지는 경향이 있음
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")), 
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \ #hyperparms의 height와 width가 동일한지 확인하고 아니면 아래 에러메세지를 띄운다.
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]] #output_filters에 channels값 3 저장
    module_list = nn.ModuleList() #nn.ModuleList()클래스의 객체인 module_list 생성
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential() # conv층 pooling 층 등 여러개의 층을 담을 수 있는 nn.Sequential()함수를 호출하여 modules에 저장

        if module_def["type"] == "convolutional": # 만약 moudel_def의 "type"에 해당하는 value 값이 "convolutional"이면 조건문을 시행한다.
            bn = int(module_def["batch_normalize"]) #배치 정규화를 통해 학습을빠르게 해준다.
            filters = int(module_def["filters"]) #convolution 계산에 사용되는 filtter의 값(갯수)을 대입한다.
            kernel_size = int(module_def["size"]) #convolution laye의 filtter의 값(갯수)을 대입한다.
            pad = (kernel_size - 1) // 2 #패딩의 사이즈를 정의한다.
            modules.add_module(  #현재 모듈에 child module(Conv2d 객체의 모듈)을 추가해준다.
                f"conv_{module_i}",
                nn.Conv2d( 
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]), 
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)) #batch normalization이 0이 아닌경우 nn.BatchNorm2d 함수를 호출하는데
                #이때 filters, momentum,eps를 대입한다. 
            if module_def["activation"] == "leaky": # 만약 활성화 함수가 leaky일 경우 nn.LeakyReLu함수를 호출한다.
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1)) #leayRelu를 수행하는 모듈을 modules에 추가, x의 범위는 0부터 0.1까

        elif module_def["type"] == "maxpool": # 만약 module_def의 value 값이 maxpool인 경우 ,각각의 filter 크기, stride 크기 설정
            kernel_size = int(module_def["size"])#kernel_size에 module_def의 "size"에 해당하는 value 값 정수형으로 변환 후 할당
            stride = int(module_def["stride"]) # stride에 module_def의 "stride"에 해당하는 value 값 정수형으로 변환 후 할당
            if kernel_size == 2 and stride == 1: #만약 필터 크기가 2 또는 stride 크기가 1인경우 nn.ZeroPad2d를 호출해서 패딩을 해준다. 
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2)) # 위에서 정의한 값들을 갖고 nn.Maxpool2d 객체 maxppol 생성 
            modules.add_module(f"maxpool_{module_i}", maxpool) # modules 리스트에 maxpool 객체 추가

        elif module_def["type"] == "upsample": #type형이 upsample인 경우 Upsample을 호출한다. 
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest") #Upsample 객체 생성, 생성자에 module_def의 "stride"에 해당하는 value값, mode=nearest
            modules.add_module(f"upsample_{module_i}", upsample) # modules 리스트에 upsample 객체 추가

        elif module_def["type"] == "route":  #type 형이 route 일 경우 layers의 값만큼 이전 단계로 가서 feature map을 가져온다. 
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", nn.Sequential()) # modules 리스트에 Sequential 객체 추가

        elif module_def["type"] == "shortcut": #type 형이 short cut일경우 ResNet의 아이디어를 이용해서 shortcut이라는 nn.module을 추가해준다.
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo": #type 형이 yolo일 경우 anchorbox를 생성후 classes의 갯수를 받아주고 img_size 및 ignore_threshold 값을 받아온다.
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors which is used for making bounding box
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            ignore_thres = float(module_def["ignore_thresh"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size, ignore_thres) #YOLOLayer 객체를 생성하는데 생성자 안에 위에서 정의한 값들을  생성자 매개변수를 넣는다.
            modules.add_module(f"yolo_{module_i}", yolo_layer)# modules 리스트에 객체 yololayer를 추가
        # Register module list and number of output filters
        module_list.append(modules)#module_lsit에 modules 추가
        output_filters.append(filters)#output_filters 리스트에 fillters 값 추가

    return hyperparams, module_list # 마지막으로 hpterparms dict형과 , module_list를 반환한다.
class Upsample(nn.Module): #Upsample class, nn.Module을 상속한다. 
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"): #scale_factor,mode 를 지정
        super(Upsample, self).__init__() #nn.moudle의 생성자 호출
        self.scale_factor = scale_factor # scale 인자 저장 yolo에서는는 3개
        self.mode = mode # 인자로 입력받은 mode를 변수로 설정

    def forward(self, x): #interpolate 보간법 함수를 호출하여 forward propagation을 한다. 
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module): #YOLOLayer nn.Module 상속
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_size, ignore_thres): #anchor의 갯수, classes의 갯수,img_size, ignore_thres의 갯수를 받아와서 생성자 함수를 만든다.
        super(YOLOLayer, self).__init__() #nn.module을 상속하므로 nn.module의 생성자도 호출한다.
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss() #meansquarederror 객체 생성
        self.bce_loss = nn.BCELoss()#bianry cross enthropy loss 객체 생성
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1) # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2) #pytorch의 view 함수를 이용하여 tensor의 shape를 바꿔준다.
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.img_size = img_size
        self.stride = None

    def forward(self, x): # forwardpropagtion을 하는 함수
        stride = self.img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #x의 차원을 transpose한다

        if not self.training:  # inference ,추론하는 단계
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(x.device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            y = y.view(bs, -1, self.no)

        return x if self.training else y

    @staticmethod
    def _make_grid(nx=20, ny=20): # gridsell을 만든다. 
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module): #nn.module 상속하는 Darknet 클래스
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416): 
        super(Darknet, self).__init__() #nn.module 생성자 호출
        self.module_defs = parse_model_config(config_path) # parsemodel_config 가 yolov3.cfg 파일의 내용을 input으로 받아 module_def 생성  module_defs 리스트에는 yolov3.cfg 안의 내용들이 담겨있다.
        self.hyperparams, self.module_list = create_modules(self.module_defs)#create_modules를 호출하여 self.hyperparams, self.module_list에 각각 값을 할당한다. 
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x): # Darknet class의 forward propagation을 수행하는 method  #layer_output 리스트와 ,yolo_outputs 리스트에 for문을 돌려 module_defs와 module_list의 값을 집어넣는다.
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:#type형이 convolutional,upsample,maxpool일 경우 모듈을 생성한다.
                x = module(x)
            elif module_def["type"] == "route": #type형이 route인 경우 torch.cat실행해서 x에 할당
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":#type 형이 short 인 경우 
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1) #yolo ouput을 리턴한다.

    def load_darknet_weights(self, weights_path): #weight가 저장된 path를 입력받아 파일안에 있는 가중치 값들을 load 해준다. 
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # 첫번째의 다섯개는 header 값들이다. 
            self.header_info = header  # 가중치를 저장하기위해 헤더를 써야한다.
            self.seen = header[3]  # header[3]= 학습하는동안 보이는 이미지의 갯수
            weights = np.fromfile(f, dtype=np.float32)  # 나머지들은 가중치이다.

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path: #만약 가중치 path 내에 "darknet53.conv.74"가 있다면 cutoff를 75로 설정해줘 75번째 layer부터 수행하도록 한다.
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Batch Normalization bias, weights 이동평균 이동 분산을 load 한다
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias 정의
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # 가중치 정의
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # 이동평균 정
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # convnet의 가중치 값을 load한다.
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1): #darknet_weights의 가중치를 저장하는 함수 
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved) #cutoffrk 1일경우 모두 저장한다.
        """
        fp = open(path, "wb")  #wb 형식으로 파일을 열어 fp에 저장후
        self.header_info[3] = self.seen # headr_info의 3번쨰 인덱스에 저장한다.
        self.header_info.tofile(fp)

        # darknet의 layer마다 각 층의 parameter들을 저장하는 것을  반복한다.
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # cov bias 값을 로드한다.
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # cov weights 값을 로드한다.
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
