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























