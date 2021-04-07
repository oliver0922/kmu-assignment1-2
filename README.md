# kmu-assignment1-2
Searching for Opensource about recognition

# 2) 자율주행 인지 관련된 2종 이상 Open Source 조사, 정리

자율주행 인지는 크게 Classification, Retrieval, Detection, Segmentation 총 네 가지로 나눌 수 있다. 그 중 이번 과제에 사용된 분야는 Classification과 Object Detection이다.

**1-1. Classification**

Classification은 자율주행 인지뿐만 아니라 다른 분야에서도 사용되는 computer vision의 기초적인 분야이다. 단순히 Classification은 어떠한 정보가 주어질 때 그 입력된 데이터를 보고 어떠한 물체인지 분류를 하는 것을 말한다. 보통 자율주행의 인지 분야에서 사용되는 Classification은 image classification 중에서 multi-label classification이고 이를 위해서 우리는 Convolutional Neural Network를 사용한다. 

![image](https://user-images.githubusercontent.com/69920975/113874206-84269b80-97f0-11eb-8778-4a306bb7c2dc.png) 

<그림 9>

<그림 9>이 인공신경망에 주어졌을 때 이를 아래와 같은 여러 개의 층에 통과를 시켜 마지막에 이 주어진 정보가 어떤 물체인지 분류한다. 예를 들어서 위와 같은 사진이 CNN Network에 주어졌을 때 고양이인 확률이 82%로 가장 값이 크므로 이 신경망은 주어진 사진이 고양이라는 결과를 도출한다. 

![image](https://user-images.githubusercontent.com/69920975/113874302-9ef91000-97f0-11eb-9c95-1688460f7901.png)

<그림 10 CNN의 구조>

인공신경망의 구조는 크게 3가지의 층으로 이루어진다.

**① Convolution 층**

Convolution 층에서는 입력 데이터를 필터가 stride만큼 움직이면서 Convolution 연산을 수행한다. 이렇게 수행을 하는 경우 신경망의 연결 관점에서 보았을 때는 Deep Neural Network에 비해 연결이 희소하고 즉 다시 말해 parameter 개수가 적다고 생각할 수 있지만 convolution 연산을 반복하면 추출된 정보가 누적되므로 사실상 parameter 개수는 줄이되 입력된 정보의 특징은 똑같이 추출하는 효과를 낼 수 있다.

![image](https://user-images.githubusercontent.com/69920975/113874348-addfc280-97f0-11eb-982e-a6fd014240c6.png)

<그림 11>

하지만 위의 <그림11>을 보듯이 Convolution 연산을 수행하면 입력 데이터의 크기가 작아진다는 단점이 있다. 즉 input data의 손실이 있을 수도 있는데 이를 방지하고자 padding 이라는 기술을 사용하는데 이는 input data 주변을 0으로 둘러싸서 convolution 연산을 수행하고도 output data의 크기가 input data의 크기와 같게 만들어 준다. 

![image](https://user-images.githubusercontent.com/69920975/113874364-b33d0d00-97f0-11eb-9ae5-512c902284e7.png) 

<그림 12 zeropadding 예>

마지막으로 convolution 층에서는 stride 값이 중요하다. stride는 input data에 겹쳐지는 filter가 움직이는 보폭을 결정하는데 이 값에 따라서 아무리 padding을 적용시켰다해도 input data에 비해 output data의 크기가 작아질 수 있다. 이제 stride 값과 padding 값을 알았으니 input data에 따른 output data의 크기를 계산할 수 있다. 아래 오른쪽 식은 각각 값들이 주어질 때 계산하는 식이다.

![image](https://user-images.githubusercontent.com/69920975/113874403-bafcb180-97f0-11eb-8ae9-a76d645b9830.png)

<그림 13 stride가 2일 때> 

![image](https://user-images.githubusercontent.com/69920975/113874417-bcc67500-97f0-11eb-9e7a-046d335369f0.png)

<그림 14 output channel을 계산하는 수식>

**② Polling 층**

 Pooling층은 두 가지 종류가 있다. Max pooling과 Average Pooling이 있는데 Max Polling은 input data에서 filter 크기만큼의 영역에서 가장 큰 값을 추출하는 것이고 Average Pooling은 마찬가지로 filter 크기만큼의 영역에서의 input data 값들의 평균을 추출하는 연산이다. Pooling 연산은 보통 input data가 조금 변형되도 값이 똑같으므로 input data에 강건하다는 특징이 있으며, 계산 횟수를 줄여준다.
 
 ![image](https://user-images.githubusercontent.com/69920975/113874484-ccde5480-97f0-11eb-8398-3f7f2e3e92ad.png)
 
 <그림 15. Max Pooling의 예시(데이터가 조금 변형되도 값은 동일하다)>
 
 ![image](https://user-images.githubusercontent.com/69920975/113874500-d49df900-97f0-11eb-86a8-6b304499e7ee.png) 
 
 <그림 16.Max pooling과 Average pooling>
 
 **③ Fully Connected Layer**
 
 ![image](https://user-images.githubusercontent.com/69920975/113874533-de276100-97f0-11eb-95d3-a76294a1888d.png)<그림 17 Fully Connected Layer>
 
 Fully Connected Layer은 말 그대로 위의 사진에서 빨간색 동그라미를 친 부분처럼 각 층의 unit 들이 1열로 늘어선 모습을 나타낸다. 만약 tensor의 shape가 Nc(채널),Nh(높이),Nw(너비) 였다면 Fully Connected Layer의 unit 개수는 Nc*Nw*Nh 개이고, 입력 network와 출력 network의 unit이 서로 빠짐없이 연결이 된다.
 
 **④ Optimizer**
 
 ![image](https://user-images.githubusercontent.com/69920975/113874577-eaabb980-97f0-11eb-8468-571449334a32.png)
 
 <그림 18. Adam optimization algorithm>
 
 이제 오픈소스에서 사용된 Optimizer에 대해 알아보자. Optimizer에는 여러 종류(SGD, NAG,NAdam)가 있는데 이번 오픈소스에서는 Adam Optimizer가 사용됐다 Adam Optimizer에는 RMS Prop 방법과 momentum hyperparameter가 같이 쓰였다. 
 
 1) RMS prop 

![image](https://user-images.githubusercontent.com/69920975/113874616-f39c8b00-97f0-11eb-9605-8cd9d9a298da.png)

<그림 19. RMS prop>

 RMS prop은 Gradient descent에서 발생할 수 있는 진동 문제를 해결하기 위해 고안되었다. 위의 <그림19>에서 편의상 세로축을 parameter b, 가로축을 parameter w라고 두고, 빨간색 지점이 loss function이 최소가 되는 지점이라고 하자. 만약 RMS prop을 적용하지 않는다면 위의 파란색 선과 같이 학습할 때마다 진동을 하다가 최소지점에 도달할 것이다. 하지만 이러한 진동은 최솟값에 도달하는데 시간이 걸리고 이는 학습 속도가 느려지는데 영향을 준다. 따라서 위의 식과 같이 backpropagation으로 얻은 loss function의 w에 대한 미분 값 dw와 loss function의 b에 대한 미분 값 db를 각각 element wise형식으로 제곱하여 Sdw와 Sdb를 계산해준다. 여기서 사용된 계산 방법은 exponentially weighted average이다. 또한  는 hyperparameter로 조절 가능한 값이다.

 마지막으로 parameter를 최신화 해줄 때 위에서 구한 Sdw와 Sdb를 각각 계산식에 나눠준다. 위의 예시와 같은 경우 Sdw를 점점 작게 만들고 Sdb를 점점 크게 만든다. 그 결과 dW는 크게 db는 작게 만들어서 세로축으로의 학습 속도는 느리게 가로축의 학습 속도는 빠르게 만들어서 진동을 줄여 최솟값에 더 빨리 도달하도록 만든다. 
 
 2) Momentum
 
 ![image](https://user-images.githubusercontent.com/69920975/113874679-031bd400-97f1-11eb-9eea-f742cf72ab69.png)
 
 <그림 20. Momentum>
 
 Momentum도 RMSP와 비슷한 방식으로 학습속도를 빠르게 만들기 위해 사용되는 hyperparameter이다.
 RMSP와 마찬가지로 parameter을 최신화 할 때 dW와 db 대신 iteration 마다 계산한 Vdw,Vdb을 대입한다.
 이 결과 gradient descent 절차를 smooth하게 만들어 준다.


**⑤ Drop out**

Drop out은 overfiiting 즉, 모델이 학습 데이터만 학습을 하여서 처음보는 test dataset을 보면 성능이 떨어지는 현상을 막기위해 사용하는 기법이다. 

![image](https://user-images.githubusercontent.com/69920975/113874768-1890fe00-97f1-11eb-9738-dcba1a209584.png)

<그림 21. Drop out>

위의 사진에서 왼쪽은 표준 신경망이고 오른쪽은 dropout을 적용한 신경망이다. epoch마다 확률을 정의해 그 확률 값만큼 한 층의 unit들을 임의로 제거한다. 이렇게 하면 linear한 효과를 줄 수 있으며 overfitting을 방지해 새로운 dataset을 보고도 좋은 결과를 낼 수 있다.

## 1-2.실습

실습은 Tensorflow에서 제공하는 image classification tutorial을 사용하였고 Google colab에서 진행했다.

구성은 다음과 같다. (코드 관련된 설명은 사진 내의 주석으로 있다.)
①Data Download and visualize

②Pre-process Dataset

③build model

④train model

⑤modify model and repeat ③~④ then test

시작에 앞서 tutorial에 필요한 package들을 import한다.

![image](https://user-images.githubusercontent.com/69920975/113874860-2d6d9180-97f1-11eb-98d1-ca855ab51a93.png)

**①Data Download and visualize(dataset은 기본으로 주어지는 꽃 데이터셋을 사용하였다)**
![image](https://user-images.githubusercontent.com/69920975/113874876-3199af00-97f1-11eb-92e6-193838747bcd.png)
![image](https://user-images.githubusercontent.com/69920975/113874900-378f9000-97f1-11eb-9678-fe0031843f82.png)
![image](https://user-images.githubusercontent.com/69920975/113874916-39f1ea00-97f1-11eb-956a-e18ae4ec4c69.png)
![image](https://user-images.githubusercontent.com/69920975/113874929-3bbbad80-97f1-11eb-8035-01a2bcd38b8f.png)

**②Pre-process Dataset**

![image](https://user-images.githubusercontent.com/69920975/113874990-4a09c980-97f1-11eb-90e2-930df3c3f5b7.png)
![image](https://user-images.githubusercontent.com/69920975/113875026-51c96e00-97f1-11eb-91bd-75475735208d.png)
![image](https://user-images.githubusercontent.com/69920975/113875045-555cf500-97f1-11eb-8f2d-c4679bb53c01.png)

**③build model**

![image](https://user-images.githubusercontent.com/69920975/113875082-5ee65d00-97f1-11eb-80e8-4d495b9b55ad.png)
![image](https://user-images.githubusercontent.com/69920975/113875092-6148b700-97f1-11eb-8cc1-515c77e4ef5a.png)

**④train model**

![image](https://user-images.githubusercontent.com/69920975/113875116-673e9800-97f1-11eb-9d55-0f10efe03e07.png)

![image](https://user-images.githubusercontent.com/69920975/113875128-69a0f200-97f1-11eb-8e40-def23c0bcbf0.png)


결과 그래프를 살펴보면, training accuracy는 1에 가깝도록 굉장히 높지만 validation accuracy는 그에 비해 상당히 낮은 편이다. loss 그래프도 마찬가지로 살펴보면 training loss는 계속 낮아지는 반면 validation loss는 증가함을 알 수 있다. 따라서 model이 training dataset에만 잘 작동하고 처음보는 validation에서는 예측을 잘 못하는 것을 알 수 있다. 즉 overfitting이 일어난 것이다. 

 따라서 overfitting을 방지하기 위해 data augmentation과 drop out을 사용할 것이다. data augmentation은 간단히 말해 이미 존재하는 training dataset을 자르거나 확대 또는 뒤집어서 data양을 늘리는 것을 의미한다.
 
 **⑤modify model and repeat ③~④ then test(Drop out data augmentation)**
 ![image](https://user-images.githubusercontent.com/69920975/113875196-79203b00-97f1-11eb-9acb-f5e627da8a44.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875211-7c1b2b80-97f1-11eb-80b0-8f253aa4ef11.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875185-77ef0e00-97f1-11eb-9109-36e3c2fc3ec9.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875227-7e7d8580-97f1-11eb-81b3-6259efced091.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875251-850bfd00-97f1-11eb-9eac-acf2c2559767.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875258-86d5c080-97f1-11eb-91de-5f8841485100.png)
 ![image](https://user-images.githubusercontent.com/69920975/113875270-889f8400-97f1-11eb-8360-701af2e6ad6d.png)
 
 Drop out과 Data augmentation을 하니 이전과 다르게 validation set의 성능도 training set을 갖고 학습했을 때와 차이가 많이 줄어든 모습을 관찰할 수 있다. 하지만 여전히 training set의 accuracy가 validation set의 accuracy보다 높으므로 overfitting을 완전히 벗어났다고 판단하기는 힘들다. 따라서 다른 hyper parameter를 tunning하거나, epoch수를 늘려 학습횟수를 늘리는 등 다른 작업이 필요하다.
 
  마지막으로 model이 이전에 한번도 학습해보지 않은 test dataset을 가져와서 classification을 하도록 실행을 시켜본다. 여기서 가져온 testset은 왼쪽 그림을 보는 것처럼 해바라기이며 아래 결과값을 보면 99.95%의 확률로 해바라기라는 것을 분류해냈다.

![image](https://user-images.githubusercontent.com/69920975/113875290-8e956500-97f1-11eb-8d23-602b563f3dc7.png)





## 2-1.Object Detection

Object detection은 오늘날 자율주행 자동차에서 많이 사용되는 인지 분야이며 물체를 분류하는 Clasification과 물체의 위치를 찾아내는 Localization을 둘 다 해내는 분야이다.

이제까지 나온 Object detection에 사용되는 architecture는 여러 가지가 있는데, 이제까지 나온 모델들을 크게 One-stage detector과 Two stage-detector로 아래 <그림1>과 같이 두 가지로 나눌 수 있다.


![image](https://user-images.githubusercontent.com/69920975/113877395-a1a93480-97f3-11eb-9b01-664601592375.png)

![image](https://user-images.githubusercontent.com/69920975/113877406-a40b8e80-97f3-11eb-9990-68dcf478608d.png)

<그림 1. One Stage Detector(위)와 Two Stage Detector>

간단히 말해 One Stage Detector는 Localization과 Classification을 동시에 수행하는 모델이며 대표적인 모델로는 SSD계열과 YOLO계열 등이 있다. 반면 Two Stage Detector는 Localization을 한 후 Classification을 순차적으로 진행하며 대표적인 모델로는 R-CNN계열의 모델들이 있다. One Stage Detector는 Localization과 Classification을 동시에 수행하므로 속도가 빠르지만 정확도가 낮다는 단점이 있으며, Two Stage Detector는 One Stage Detector에 비해 속도가 느리지만 정확도가 높다는 장점이 있다. 실제 자율주행 차량은 실시간으로 차량 주변의 상황을 빠르게 인식해야하므로 One Stage Detector가 주로 사용되고 따라서 이번 과제에서 널리 알려져있는 One Stage Detector 모델 YOLOv3를 사용하였다.

**① YOLOv3** 

YOLOv3 모델을 설명하기에 앞서 Object Detection을 이해하기 위해서는 IOU(Intersection Over Union)과 mAP(mean Average Precision)에 대한 이해가 필요하다.

1) IOU(Intersection Over Union)

IOU란 Intersection Over Union의 줄임말로 Ground Truth(정답)에 해당하는 사각형과 Bounding Box(예측값)의 교집합을 두 사각형의 합집합으로 나눈 것으로 bounding box를 얼마나 잘 예측하였는지 즉, Object Detector의 평가를 위한 척도이며 IOU가 아래 <그림2>와 같이 클수록 Ground Truth와 Prediction을 하는 bounding box가 많이 겹치므로 성능이 좋은 Object Detector이다.

![image](https://user-images.githubusercontent.com/69920975/113877525-bf769980-97f3-11eb-85a3-72a1f914cb9b.png)

 <그림 2>

2) mAP(mean Average Precision)

mAP(mean Average Precision)을 이해하기 위해서는 우선 Precision(정밀도)와 Recall(재현율)에 대한 이해가 필요하다. 이해를 위해 아래 <표1>을 살펴본다.

![image](https://user-images.githubusercontent.com/69920975/113877577-cd2c1f00-97f3-11eb-853c-008cf00c15d8.png)

<표1 Confusion Matrix >

![image](https://user-images.githubusercontent.com/69920975/113877606-d3220000-97f3-11eb-903b-58baf9b0a343.png) ----(1)

Precision(정밀도)은 위 수식 (1)과 같이 나타내며 Classifier가 어떠한 물체를 검출하였을 때 얼마나 옳은 검출을 했는지 나타내는 지표이다.

![image](https://user-images.githubusercontent.com/69920975/113877644-dae1a480-97f3-11eb-9c0e-ac566e13da39.png)----(2)

Recall(재현율)은 위의 수식 (2)와 같이 나타내며 Classifier가 마땅히 검출해내야하는 물체들 중에서 제대로 검출된 것의 비율을 의미한다.

일반적으로 Precision을 올리면 Recall이 줄어들고 그 반대도 마찬가지이다. 이를 Precision Recall Trade off라고 하며 Precision과 Recall을 각각 x축 y축에 두고 그린 Precision-Recall 곡선은 아래 <그림 3>과 같다. 

![image](https://user-images.githubusercontent.com/69920975/113877694-e92fc080-97f3-11eb-8a5a-d56aad2458da.png)

<그림 3>

 위 개념을 사용하여 AP라는 개념을 정의할 수 있다. AP는 Average Precision의 약자로 위 그래프의 면적을 나타낸다. AP는 다음과 같이 계산한다. 최소 0% Recall에서 얻을 수 있는 최대 Precision, 다음 10%, 20% 이런 식으로 100% Recall에서의 최대 Precision을 계산한 후 평균을 내준다. 만약 두 개 이상의 class가 있을 때는 각 class에 대하여 AP를 계산 후, class의 개수로 나눠준다. 이것이 mAP이다.
 
 하지만 Object Detector에서 사용하는 mAP는 조금 더 복잡하다. 예를 들어 모델이 정확한 class를 탐지했지만 위치가 잘못되었다면(즉 bounding box 밖으로 객체가 벗어나면) 이를 올바른 예측으로 포함시키면 안된다. 따라서 위에서 정의한 IOU 개념을 사용한다. 만약 IOU가 0.5보다 크며 예측한 class가 맞으면 이때를 올바른 예측으로 간주하며 이를 mAP@0.5로 표현한다. COCO와 같은 대회에서는 여러 IOU값에서 mAP를 계산 후 이를 평균을 내준다.
 
3) YOLOv3의 특징

YOLOv3는 이미 나온 YOLO 모델의 성능을 약간 좋게 향상 시킨 모델이며 기존의 YOLO모델의 비해 큰 차이는 없다. 

**① Bounding Box Prediction**

![image](https://user-images.githubusercontent.com/69920975/113877822-06648f00-97f4-11eb-978d-aaff19dc2628.png)

![image](https://user-images.githubusercontent.com/69920975/113877838-095f7f80-97f4-11eb-93c4-18888624b41f.png)
![image](https://user-images.githubusercontent.com/69920975/113877846-0bc1d980-97f4-11eb-9c0f-4bea8cde68cd.png)

YOLO9000과 마찬가지로 YOLOv3는 dimension cluster를 anchor box로 사용한다. 이 신경망은 각 bounding box의 4가지 좌표축을 예측한다. bounding box의 4가지 좌표는 sigmoid 함수, 및 exponential 함수를 사용하여 위의 수식과 같이 구한다. 또한 bounding box를 찾는 것은 regression의 문제이니 data를 training 시키는 동안 squared error loss를 사용한다. 즉 미리 anchor 박스를 정의해놓고 regression방법을 사용하여 achorbox를 얼마나 움직일지 예측하는 것이다.


YOLOv3는 logistic regression을 사용하여 각 bounding box의 objectness score(confidence score)를 예측한다. 만약 bounding box prior가 다른 어떤 bounding box prior보다  ground truth와 겹친다면 objectness score가 1이 되야한다. 


0.5를 threshold 값으로 사용하며, Faster R-CNN과 YOLOv3는 각 ground truth object에 대해 하나의 bounding box만을 할당한다.

**② Class Prediction**

보통 다른 모델들은 80개의 class(COCO기준)에 대해 softmax function을 사용하는데에 반해 YOLOv3는 sigmoid functinon을 사용해 mult-ilabel bianary classfication을 이용한다.  

**③ Predictions Across Scales**

 YOLOv3의 가장 큰 특징은 3개의 서로 다른 scale로 bounding box를 예측한다는 것이다. 또한 YOLOv3는 각 scale에 대해 3개의 box를 예측한다. 즉 9개의 box를 사용한다. 따라서 tensor의 크기는 N*N*(3*(4+1+80))이다. (4= bounding box의 정보 개수, 1= confidence score 또는 objectness score, 80=COCO dataset의 class 개수) 이를 나타내면 아래 그림과 같다. 
 
 ![image](https://user-images.githubusercontent.com/69920975/113878007-2eec8900-97f4-11eb-9cec-bf15448613e4.png) 
 
 <그림4>

**③ Anchor Box**

YOLOv3는 데이터 셋을 분석하여 k-means clustering을 사용하여 anchor박스를 정의한다. 즉 9개의 clusters와 3개의 sclale을 임의로 anchor box dimnesion을 할당한다.

**④ Feature Extraction**

Detection에는 back bone이 되는 CNN이 들어가야하는데, YOLOv1에서는 VGG net을 사용했으며 YOLOv2에서는 Darknet-19을 사용하고 YOLO v3에서는 새로운 모델 Darknet-53을 사용한다. 아래 구조를 보면 알 수 있듯이 YOLO v3는 3*3, 1*1 의 convolutional layer를 사용했지만, short connectino 때문에 네트워크 크기가 커졌다. 동일 환경에서 실험 결과 Darknet-53은 기존 Darknet-19 보다 강력하지만 ResNet 모델들보다 조금 더 효율적이다. 아래 표를 보면 Darknet-53은 BFLOP/s(floating point operation/seconds) 부문에서 좋은 성능을 보인다. 이는 Darknt-53이 GPU를 잘 활용한다는 것을 의미한다. 

**⑤ YOLOv3 Architecture**

![image](https://user-images.githubusercontent.com/69920975/113878113-475ca380-97f4-11eb-9760-1d7f1ed89860.png)

위 그림은 YOLOv3의 전체적인 구조를 나타낸 그림이다. Darknet-53을 기본 구조로 갖고 있으며 순서대로 scale1,scale2,scale3가 있다. 여기서 scale1은 resolution이 제일 작으므로 큰 물체를 찾고 scale2는 중간물체 마지막으로 scale3은 resolution이 가장 크므로 제일 작은 물체를 찾아낸다. 또한 초록색으로 표시되는 것은 ResNet에서 사용되는 feature pyramid network를 나타낸 것으로, 위치 정보가 점점 올라갈수록 사라지는 것을 다시 역으로 더해 위치정보에 대한 성능을 높인다. 

**⑥ Training**
 YOLOv3는 full image를 사용하며 class에 background class라는 것을 포함시키지 않아 negative mining(data unbalance)은 사용하지 않는다. 또한 multi-scale training, data augmentation, batch normalization 등 규격화를 사용한다. 
 
## 2-2. 실습

**① OpenSource**
 실습을 위해 또한 github에서 clone을 해왔는데 git의 주소는 아래와 같다.
https://github.com/eriklindernoren/PyTorch-YOLOv3.git
 
**② Dataset**
 위의 repository에서 제공하는 dataset은 coco dataset이며 train dataset만 해도 약 8만개여서 구글 colab환경에서 그만큼 많은 dataset을 이용하는 것은 디스크 용량 문제 때문에 불가능했다. 따라서 약간의 customize를 하여서 모델을 학습시키길 결정했다. 모델의 학습에 사용된  dataset은 roboflow에서 label이 이미 완료된 UDACITY의 self driving car 수업의 dataset들을 사용하였다. dataset은 image와 label 각각 15000개이며, ipynb를 참고하면 나오겠지만 sklearn의 train_test_split 함수를 사용하여 train set와 validation data set을 각각 14700개와 300개로 분할하였다. image는 jpg 파일 형식이며 label된 정보는 txt 파일에 담겨있다. 
 
 ![image](https://user-images.githubusercontent.com/69920975/113878286-6eb37080-97f4-11eb-8e3c-e795b6e47e50.png)
 
 <Dataset의 image의 예시>
 
 ![image](https://user-images.githubusercontent.com/69920975/113878313-74a95180-97f4-11eb-977d-47c8b0f7aee6.png)
 
 <Dataset의 label.txt의 예시>
 
 학습을 하기 위해서 주의할 점이 있는데 image data의 파일명과 그에 상응하는 정보가 담긴 label data의 파일명이 같아야 한다는 것이다.

위의 예시의 각 한 줄은 bounding box 하나에 대한 정보가 담겨있다. 
앞에서부터 총 5개의 숫자가 있는데 각 숫자가 의미하는 것은 순서대로 다음과 같다.
1 0.6325520833333333 0.5016666666666667 0.10677083333333333 0.06
첫 번째 숫자 1 : 각 class의 label을 의미한다, 여기서는 2번째 class car
두 번째 숫자~ 다섯 번째 숫자: 0과 1사이의 값이며 bounding box의 x,y,w,h를 의미한다.

모델이 분류한 class는 총 11개가 있으며 0부터 인덱스를 세야한다. 11개의 class는 아래와 같다. 

0:biker
1:car
2:pedestrian
3:trafficLight
4:trafficLight-Green
5:trafficLight-GreenLeft
6:trafficLight-Red
7:trafficLight-RedLeft
8:trafficLight-Yellow
9:trafficLight-YellowLeft
10:truck

또한 label.txt에 담긴 줄의 수 만큼 bounding box를 생성했고, 이는 객체를 그 개수만큼 객체를 인식했다는 것을 알 수 있다.




**③구성**

위의 github에서 repository를 clone 하면 아래와 같이 directory들이 구성된다.

![image](https://user-images.githubusercontent.com/69920975/113878433-8a1e7b80-97f4-11eb-9fc5-2ce6bb30a437.png)

coco.data: cocodataset의 class 개수(80), train dataset의 path, valid dataset의 path, coco data set의 class가 적힌 파일 path 등이 적혀있다.
create_custom_model.sh: yolov3_custom.cfg 파일을 만들기 위해서 작성된 linux 명령어가 담긴 파일이다.
yolov3-custom.cfg: yolov3에 custom dataset을 적용시킬 때 필요한 class의 개수, hyper parameter 및 yolo의 전체적인 골격이 아래 사진과 같이 나타나있다. 

![image](https://user-images.githubusercontent.com/69920975/113878466-9276b680-97f4-11eb-9236-32f974b05af0.png)
![image](https://user-images.githubusercontent.com/69920975/113878487-99052e00-97f4-11eb-98ab-08370847f1e8.png)


yolov3_tiny.cfg, yolov3.cfg: 마찬가지로 yolov3를 사용하기 위해 필요한 hyperparameter 값들이 담겨있으며 전체적인 구조가 담겨있고, yolov3.tiny.cfg는 모델을 조금 더 작은 것을 쓰고자 할 때 사용된다. 하지만 학습 효과는 그리 좋지 못하다.

data 폴더: 아래 사진 참조

![image](https://user-images.githubusercontent.com/69920975/113878523-a28e9600-97f4-11eb-9e43-824b46b2ee85.png)

yolov3를 학습시킬 dataset들이 저장되는 폴더이며 custom 폴더에 사용자의 dataset 파일들을 image 폴더와 label 폴더에 각각 나누어서 넣는다.

logs 폴더: 학습 시키는 동안의 log 기록들이 저장되어있는 폴더이다.
output 폴더: yolov3 모델에 임의의 data를 적용했을 때 아래처럼 bounding box가 생성되어 나오는 사진들이 담긴 폴더이다.

![image](https://user-images.githubusercontent.com/69920975/113878549-a9b5a400-97f4-11eb-8408-9746e488acb0.png)

utils 폴더: train.py, test.py 등 다른 모듈들을 실행하고자 할 때 필요한 package들이 담겨있는 폴더이다.

weights 폴더: 이미 학습된 파라미터들이 담겨있는 yolov3. weights 파일, darknet backbone이 들어있는 darknet53.conv.74 파일 등이 담겨있다.

train.py: model을 학습시키고자 할 때 실행시키는 파일로 실행 중간에 test.py 내의 evaluate 함수를 호출하여 model을 validation set에 대한 평가하며 models.py 내부에 있는 여러 class 들을 가져와 사용한다. 코드 내용은 아래 사진과 같이 주석을 달아놓았다. 

test.py: 학습된 모델을 평가하고자 할 때 사용되는 파일이고 코드와 코드 설명은 아래와 같다.

models.py: module을 만들고 YOLO class와 Darknet class를 정의해놓은 파일이다. 코드설명은 아래와 같다.

requirement.txt: 각 python 모듈을 실행하기 위해서 필요한 package 들에 대한 정보가 담겨있으며 이를 실행하면 package 들이 google colab 환경에 설치된다.













