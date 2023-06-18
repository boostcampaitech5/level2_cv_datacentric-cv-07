# **학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회**

## Project Overview
### 프로젝트 목표
 - 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나이다. 본 대회는 이러한 OCR 기술을 학습 데이터 추가 및 수정을 통해서만 개선하는 것을 목표로 하였다. 이에 데이터를 제외한 각종 코드에 제약사항이 있었고 오직 데이터 중심의 접근법을 통해서만 모델의 성능 향상을 추구하였다.

<br>

### 개발 환경
  - 팀 구성 및 컴퓨팅 환경: 5 인 1 팀, 인당 V100 GPU 및 aistages 서버
  - 개발 환경: VSCode, Jupyter Notebook
  - 개발 언어: Python
  - 협업 툴 및 기타: Slack, Notion, Github, Wandb



<br>
  
## Team Introduction
### Members
| 고금강 | 김동우 | 박준일 | 임재규 | 최지욱 |
|:--:|:--:|:--:|:--:|:--:|
|<img  src='https://avatars.githubusercontent.com/u/101968683?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/113488324?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/106866130?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/77265704?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/78603611?v=4'  height=80  width=80px></img>|
|[Github](https://github.com/TwinKay)|[Github](https://github.com/dwkim8155)|[Github](https://github.com/Parkjoonil)|[Github](https://github.com/Peachypie98)|[Github](https://github.com/guk98)|
|twinkay@yonsei.ac.kr|dwkim8155@gmail.com|joonil2613@gmail.com|jaekyu.1998.bliz@gmail.com|guk987@naver.com|

<br>

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 고금강_T5011 | - 실험 간편화를 위한 Configuration 구현 <br> - 실험 자동화를 위한 Auto Trainer 구현 <br> - Random Seed Per Epoch 코드 구현 <br> - 다양한 Augmentation 실험 |
| 김동우_T5026 | - 추가 학습 데이터 수집 및 Roboflow를 통한 데이터 Annotation 수행 <br> -평가 Metric(deteval) 및 Inference 시각화 코드 구현 <br> - Normalize, Json concat 등 각종 유틸 구현 |
| 박준일_T5094 | - 추가 학습 데이터 수집 및 Roboflow를 통한 데이터 Annotation 수행 <br> - Pickle을 사용한 모델 학습 시간 단축 구현 <br> - 다양한 하이퍼파라미터(LR, Epochs, Size, etc..) 실험 |
| 임재규_T5174 | - Early Stopping, CutMix 구현 <br> - Roboflow를 통한 추가 데이터 Annotation 수행 <br> - Data Augmentation 시각화 구현 <br> - Train 스크립트에 wandb 재개 기능 추가 |
| 최지욱_T5219 | - Custom bbox ensemble 방법(WBF with hard voting) 구현 <br> - 별도 수집한 저해상도 이미지에 대해 Super resolution을 적용하여 해상도를 개선 <br> - 학습된 모델의 추론 결과에 대한 정성적인 사후 분석 수행 <br> - Annotation format간 양방향 변환(UFO <-> COCO) 구현 |

<br>

## Procedure & Techniques

### Data
  -  Baseline으로 받은 초기 진단서 데이터는 100개였고, 이후 수작업으로 라벨링한 201개의 데이터를 추가 제공 받았다. 각 데이터마다 Annotation된 글자가 많았지만 그래도 데이터수가 부족하다고 판단해 학습 데이터를 추가하기로 결정했다. 이번 대회의 경우 모델이 Transcription을 제외한 글자 영역(bbox 4개 좌표)만을 Output 값으로 도출하기 때문에 Custom Dataset을 만들기 용이했다. 따라서 우리 팀은 별도로 추가 데이터를 확보하고 Roboflow를 사용하여 라벨링을 진행하였다.
  
  
<br>

### Additional Data
  -  제공된 301개 진단서 데이터로는 모델 학습하는데 부족하다고 생각했기 때문에 Roboflow를 통해 25개의 추가 진단서 데이터를 생성해 총 326개 데이터로 학습했다. 하지만 모델 학습이 끝나고 테스트 데이터에 대해 Inference 결과를 시각화 했을 때 아래 사진과 같이 글자가 아닌 배경에 라벨링 되거나, QR 코드 내부 또는 글자 인식이 힘든 도장에 라벨링 되는 경우가 있었으며, 90도 기울어진 가로쓰기 글자는 제대로 라벨링이 되지 않는 경우가 있었다.
  -  따라서 글자가 아닌 것들을 인식하지 않도록 아래 그림과 같이 Annotation을 수행하지 않은 배경 사진, QR 코드, 글자 인식이 힘든 도장 등 글자 검출에 방해가 되는 요소가 포함된 사진을 학습데이터에 추가했다. 또한, 추가로 수집된 데이터의 경우 대부분 해상도가 기존에 부여된 이미지보다 현저히 낮아서 글자의 형태가 명확하지 않다는 문제점이 있었다. 따라서 추가로 수집된 학습 이미지에 대해서 해상도를 고해상도로 개선하여 학습에 이용하고자 하였다. 개선 방법으로 Super Resolution 방법 중 하나인 EDSR을 사용하였고, 추가 수집된 데이터에 대해 사전 학습된 EDSR을 이용해 전처리 후 학습에 이용하였다.

<br>

### Ensemble

<br>

  - 이번 글자 검출 과제에서 기존 Object Detection에서 이용하는 Weighted Boxes Fusion과 같은 방법을  그대로 이용하기에 몇 가지 어려운 측면들이 있어 과제에 적용 가능하도록 과정을 Custom 하였다.

  - 첫 번째로 직사각형 형태의 Box들을 통합하는 WBF와 달리 이번 글자 검출에서는 직사각형을 보장하지 않는 Polygon 지점(UFO format)을 찾아내는 모델이었다. 그러나 대부분의 글자는 휘어지거나 사다리꼴이 아닌 직사각형 형태의 Box로 Label 되어있다는 것을 확인하였기 때문에 UFO 형식의 사각형에 외접하는 직사각형(COCO)의 형태로 변환하여 WBF를 수행한 뒤 결과물은 다시 UFO 형식으로 변환하였다.

  - 두 번째로는 대회 특성상 Confidence Score를 이용할 수 없다는 한계점이 있었다. Confidence Score를 이용할 수 있는 경우에는 임계치를 정해두고 낮은 Confidence Score를 갖는 False Positive 검출을 억제할 수 있으나, 이번 프로젝트에서는 Confidence Score를 이용할 수 없었기에 False Positive 검출이 지나치게 많아지고 Ensemble 시에 Precision이 감소하는 어려움이 있었다. 이를 보완하고자 Ensemble을 구성하는 각 모델들의 결과 중 N개 모델 이상에서 검출된 Box만을 신뢰성이 있다고 간주하고 그 Box들만을 대상으로 WBF을 수행하도록 하였다. 설명한 알고리즘의 절차는 아래와 같다. 

    - 1. 하나의 특정 검출 D에 대하여 검출의 중심점 C(D)을 계산한다.
    - 2. 다른 모델들의 모든 검출들에 대하여 해당 중심점 C(D)을 포함하는지를 연산한다.
      - b-1. 해당 중심점 C(D)을 포함하는 bbox가 N개 이상인 경우 검출 D를 WBF 대상에 포함한다.
      - b-2. 해당 중심점 C(D)을 포함하는 bbox가 N개 미만인 경우 검출 D를 WBF 대상에 제외한다.
    - 3. 동일 모델의 다른 검출, 다른 모델의 검출들에 대해서도 1, 2 과정을 반복한다.
    - 4. 대상이 되는 bbox만을 대상으로 WBF를 수행한다.

  - 위와 같이 Hard Voting 개념을 더하여 WBF를 Custom한 결과 False Positive 검출을 크게 줄이면서 Ensemble을 수행할 수 있었고, Precision 측면에서 큰 이득을 볼 수 있었다. 최종 제출 결과에서도 Ensemble 구성 모델 결과 중 최대 F1 Score인 0.9724에서 0.9815로 큰 폭의 성능 향상을 확인하였다.


<br>

## Results

### 단일모델

| **Model** | **Dataset** | **Epoch**  | **F1-Score** | **Recall** | **Precision** |
| :--:      | :--:        |  :--:      | :--:         | :--:       | :--:          | 
| EAST      | 326         | 125        |  0.9691      | 0.9682     | 0.9700        |
| EAST      | 360         | 134        |  0.9702      | 0.9659     | 0.9746        |
| EAST      |100(+201)    | 1000(+80)  |  0.9724      | 0.9717     | 0.9731        |

<br>

### 앙상블
| **Model** | **F1-Score** | **Recall** | **Precision** |
| :--: | :--: | :--: | :--: |
| **Ensemble of Models above** |  0.9815      | 0.9818     | 0.9812      |

<br>

-  단일 모델의 경우 데이터가 많고 Epoch이 클수록 좋은 성능을 보였다. 이번 프로젝트의 경우 테스트 데이터의 분포와 학습 데이터의 분포 간 차이가 적어, Epoch을 높게 잡아도 Overfitting의 문제없이 모델의 성능이 향상되었다. 앙상블 실험에서는 표에 있는 세 가지 단일 모델을 앙상블 했을 때 가장 높은 F1 Score을 얻을 수 있었다. 단순히 성능이 가장 높은 단일 모델을 앙상블할 때 보다, 서로 다른 Augmentation을 적용했고 다른 데이터셋을 사용한 단일 모델을 앙상블했을 때 더 좋은 결과를 얻을 수 있었다.


<br>


### 최종 순위
- 🥈 **Public LB : 4nd / 19**
<img width="869" alt="스크린샷 2023-06-18 오후 10 28 58" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-07/assets/113488324/bedd196b-e026-4623-8ff8-35f92b8e8c7b">

- 🥈 **Private LB : 3nd / 19**
<img width="871" alt="스크린샷 2023-06-18 오후 10 29 56" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-07/assets/113488324/c85a0163-6c3f-4d08-afd6-8f8299c70096">
