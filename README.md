# LENet

Graph Transformer Networks (GTN) : to be trained **globally** using **gradient-based** methods so as to minimize an overall performance measure

전체적인 성능 향상을 위해 **여러 모듈**들에 대해 **Gradient 기반 학습**을 진행함

# Introduction

기존 패턴 인식 모델은 다음과 같은 형태

**The feature extractor (특징 추출기)**

: 입력 패턴을 변환하여 저차원 벡터 또는 짧은 문자열로 나타냄

1) 일치하거나 비교할 수 있고,  2) 특성을 변경하지 않는 입력 패턴

사람이 추출한 사전지식을 포함하여 작업함

**The classifier (분류기) :** 범용적이고 훈련 가능함

기존 fully-connected network의 경우 입력 데이터의 topology를 무시하고 학습함

따라서 높은 성능을 위해서는 translation(문자 위치 이동/회전 등), distortion(문자 형태 변형) 등 모든 경우에 대한 더 많은 학습 데이터가 필요함 

최근 음성 및 필기 인식 시스템의 정확도 향상은 학습 기술과 대규모 훈련 데이터 세트에 대한 의존도가 높아진 데 크게 기인한다고 주장할 수 있음

CNN은 2차원 이미지의 픽셀에서 edge나 corner와 같이 특정 부분(local feature)에 대한 특징 추출을 통해 topology 문제를 해결하고자 함

- 이를 receptive fields(수용영역)이라고 하며, **kernel(또는 filter)**를 적용한다고 표현함

이 과정을 **convolution**한다고 하며, 여기서 filter는 하나의 weight가 됨

이미지 프로세싱에서는 원하는 효과를 위해 특정한 값으로 이루어진 filter를 사용하지만, CNN에서는 해당 weight값을 학습을 통해 계속하여 업데이트하도록 함

convolution이 끝나면 filter가 적용되어 입력 데이터로부터 detect된 receptive field들의 집합인 **feature map**을 얻게 됨

이렇게 filter을 통해  다음 layer를 local하게 구성함으로써 사용되는 weight 파라미터의 수가 줄어듦

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b89faf0-c3d9-4763-987a-3a6d9b84251e/Untitled.png)

- Shared weights란, convolution할 때 **적용하는 filter(weight) 값은 변하지 않는 것**
- sub-sampling는 추출한 local feature로부터 입력된 데이터의 **topology에 영향을 받지 않는 global feature를 추출**하기 위함

해당 과정을 거치고 classification 진행

# LENet-5

- 흑백 영상을 입력으로 받음(32x32)
- 총 7개의 레이어로 구성
- Cx : convolutional layer
- Sx : sub-sampling layer
- Fx : fully-connected layer

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38ab4254-3219-4f0f-a657-4c5c0ef833c9/Untitled.png)

모든 S2 feature map을 모든 C3 feature map에 연결하지 않는 이유? 

1) 불완전한 연결 체계는 합리적인 범위 내에서 연결 수를 유지합니다. 

2) 더 중요한 것은 네트워크에서 **대칭이 깨지도록** 강제한다는 것입니다. 서로 다른 기능 맵은 서로 다른 입력 세트를 얻기 때문에 서로 다른(상보적인) 기능을 추출해야 합니다.

6개의 feature maps를 처음 6개는 연속되게 이웃하는 3개의 feature map들의 조합에서 구하고, 그다음 6개는 연속되게 이웃하는 4개 feature maps의 조합, 그 다음 3개는 sparse하게 남는 4개의 조합, 그리고 마지막 16번째는 6개를 모두 다 반영한 feature를 뽑을 수 있도록 아래와 같이 구성합니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/805533c4-fa3b-4097-a2d5-8045231058ab/Untitled.png)

CNN이 적용됨으로써 입력 데이터의 translation, distortion과 같은 topology변화에 영향을 덜 받음을 알 수 있음

- 합성곱 신경망(CNN)으로 어떻게 topology 문제를 해결할까?
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d0b714a6-6898-4866-bd8e-1229141da217/Untitled.png)
    
    물체의 형태와 방향에 따라 활성화되는 뉴런이 다름
    
    시각 피질 안의 뉴런은 일정 범위 안에서의 자극에만 활성화되는 ‘근접 수용 영역(local receptive field)’를 가지며, 이 수용 영역들이 겹쳐져 전체 시야를 이룸
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6765b9e8-23b1-48a6-a41c-54ef3e201927/Untitled.png)
    
    기존 완전 연결 신경망의 경우 3차원 이미지 데이터를 학습시키기 위해 1차원으로 평면화하는 과정에서 이미지의 공간정보가 유실됨
    
    이미지 내 거리가 가까운 픽셀들은 서로 연관을 가지게 되고, 이는 사진에서 음영, 선, 질감 등으로 표현됨
    
    CNN은 완전 연결 신경망과 달리 입력 데이터의 구조를 유지해서 입력을 받고, 전달하므로 공간정보를 잃지 않고 이미지 데이터를 이해할 수 있음
    
    입력 받은 이미지를 잘 분류하기 위해 ‘이미지에 필터를 씌워’ 특징을 찾는 과정을 합성곱 연산이라고 함
    

- 참고
    
    [LeNet - 1998 | DataCrew](https://datacrew.tech/lenet/)
    
    [합성곱 신경망(CNN) / 고양이의 눈에서 답을 얻다.](https://ardino.tistory.com/38)
    
    [[논문 요약 3] Gradient-Based Learning Applied to Document Recognition](https://arclab.tistory.com/150)
