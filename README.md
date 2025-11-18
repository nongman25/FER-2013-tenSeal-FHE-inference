# FHE Emotion Recognition Demo

PyTorch와 TenSEAL을 이용해 FER2013 얼굴 감정 인식 데이터셋을 준비하고, FHE 친화적인 얕은 CNN을 학습한 뒤 암호화된 추론을 수행하는 데모 프로젝트입니다. 정확도보다는 **엔드 투 엔드 파이프라인** 동작과 **동형암호 연산 흐름**을 확인하는 데 목적이 있습니다.

## 주요 특징
- FER2013 CSV를 불러와 `torch.Tensor` 형태로 전처리하고, 클래스 불균형을 보정하기 위한 가중치를 계산합니다.
- 평균 풀링과 선형 다항 활성화(`PolyAct(x) = a·x + b`)만을 사용하는 얕은 CNN(`FHEEmotionCNN`)을 PyTorch로 학습합니다. 이 활성화는 곱셈과 덧셈만 사용해 CKKS 스케일 폭주 없이 TenSEAL에서 재현할 수 있습니다.
- TenSEAL CKKS 컨텍스트에서 동일한 연산을 스칼라 단위로 재현하여 단일 이미지 암호문 추론 데모를 제공합니다.

## 환경 준비
```bash
cd fhe_emotion
./scripts/setup_env.sh
```
- `.venv` 가상환경을 생성하고 PyTorch, TenSEAL, Jupyter 등을 설치합니다.
- `fhe-emotion` 이름의 Jupyter 커널이 자동 등록됩니다.

## 데이터 준비 절차
1. Kaggle API 자격 증명(`KAGGLE_USERNAME`, `KAGGLE_KEY`)이 설정돼 있으면 `fhe_emotion/data/` 안에 `fer2013.csv`가 없을 때 자동 다운로드합니다.
2. Jupyter에서 `notebooks/01_prepare_fer2013.ipynb`를 실행하거나, 다음처럼 비대화식으로 실행합니다.
   ```bash
   cd fhe_emotion
   jupyter nbconvert --to notebook --execute notebooks/01_prepare_fer2013.ipynb --inplace
   ```
3. 전처리 결과는 `data/processed/` 아래에 `*_images.pt`, `*_labels.pt`, `class_weights.pt`로 저장됩니다.

## 모델 학습
- 대화식: `notebooks/02_train_plain_cnn.ipynb`에서 학습 과정을 실행합니다.
- 비대화식: 준비된 스크립트를 이용합니다.
  ```bash
  cd fhe_emotion
  ./scripts/run_train.sh
  ```
- 최적 가중치는 `models/fhe_cnn_fer2013.pt`, 입력 정규화 통계는 `models/normalization_stats.json`으로 저장됩니다.

## 암호화 추론 데모
- 단일 샘플에 대해 평문/암호문 추론 결과를 비교합니다.
- CLI 실행:
  ```bash
  cd fhe_emotion
  ./scripts/run_encrypted_infer.sh
  ```
- 노트북: `notebooks/03_tenseal_encrypted_inference.ipynb`에서 이미지 시각화와 로그를 동시에 확인할 수 있습니다.
- TenSEAL 추론 실행기:
  - `PackedEncryptedCNNRunner`(기본): TenSEAL 문서의 im2col/행렬곱 패턴을 따라 채널 전체를 하나의 CKKSVector에 패킹합니다. 회전 연산 대신 `ckks_vector.mm(plain_tensor)`으로 미리 계산한 permutation/평균풀링 행렬을 곱해 합성곱‧풀링을 수행하므로 현재 TenSEAL API(rotate 미제공)와 호환됩니다.
  - `EncryptedCNNRunner`: 픽셀별 스칼라 암호문을 사용하는 디버그용 구현입니다. `encrypted_inference_demo(..., use_packed=False)`로 호출할 수 있습니다.
- 컨텍스트는 `he/tenseal_context.py`에서 설정한 대로 CKKS(폴리 차수 8192, 모드 체인 `[60, 40, 40, 60]`, 스케일 `2**40`)이며 Galois/Relin 키를 생성해야 `mm` 기반 패킹 추론이 동작합니다. 기본 체인 생성에 실패하면 코드가 자동으로 `[40, 21, 21, 40]` 등 대체 체인을 재시도하므로 노트북/CLI 어디서나 안정적으로 컨텍스트를 만들 수 있습니다.

## 디렉터리 및 파일 설명
- `fhe_emotion/models/fhe_cnn.py` : 다항식 활성화와 평균 풀링만 사용하는 FHE 친화적 CNN 정의, TenSEAL용 파라미터 추출 도우미 포함.
- `fhe_emotion/he/tenseal_context.py` : CKKS 컨텍스트 생성, 벡터 암복호화, 컨텍스트 직렬화 유틸리티.
- `fhe_emotion/he/fhe_inference.py` : PyTorch 가중치 로딩, 스칼라 기반 TenSEAL 연산(합성곱, 평균 풀링, 선형계층, PolyAct) 구현, `encrypted_inference_demo` 제공.
- `fhe_emotion/notebooks/*.ipynb` : 01 데이터 준비, 02 모델 학습, 03 TenSEAL 추론 흐름을 단계별로 재현하는 노트북.
- `fhe_emotion/scripts/setup_env.sh` : 프로젝트 의존성 설치 및 Jupyter 커널 등록.
- `fhe_emotion/scripts/run_train.sh` : 학습 노트북 비대화식 실행.
- `fhe_emotion/scripts/run_encrypted_infer.sh` : TenSEAL 추론 모듈 실행.
- `fhe_emotion/data/processed/` : 전처리 후 저장되는 텐서 및 클래스 가중치.

## 참고자료
- 데이터셋: [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- FHE 패턴: [smile-ffg/he-man-tenseal](https://github.com/smile-ffg/he-man-tenseal)

필요한 부분(학습 epoch 수, 컨텍스트 파라미터 등)을 조정하면 더 빠른 실험이나 정확도 개선을 쉽게 시도할 수 있습니다.
