# FHE Emotion Recognition Demo

PyTorch와 TenSEAL을 이용해 FER2013 얼굴 감정 인식 데이터셋을 준비하고, FHE 친화적인 얕은 CNN을 학습한 뒤 암호화된 추론을 수행하는 데모 프로젝트입니다. 정확도보다는 **엔드 투 엔드 파이프라인** 동작과 **동형암호 연산 흐름**을 확인하는 데 목적이 있습니다.

## 주요 특징
- FER2013 CSV를 불러와 `torch.Tensor` 형태로 전처리하고, 클래스 불균형을 보정하기 위한 가중치를 계산합니다.
- **Wide 1-Conv 아키텍처**: Stride 3 합성곱(16 채널, 7x7 커널) + Square 활성화 + 2-layer FC를 사용하는 얕은 CNN(`FHEEmotionCNN`)을 PyTorch로 학습합니다.
  - Batch Normalization 제거: FHE 추론 단순화를 위해 BN 완전 제거
  - Multiplicative Depth 최소화: Conv → Square → FC → Square (총 4단계)
  - CKKS 슬롯 효율: Stride 3 사용으로 16×196=3,136 슬롯만 필요 (32768 poly_modulus_degree에 여유롭게 수용)
- TenSEAL CKKS 컨텍스트에서 im2col 기반 패킹 추론을 통해 암호화된 이미지에 대한 감정 분류를 수행합니다.

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
- 학습 설정: 50 epochs, Adam optimizer, CrossEntropyLoss with class weights
- 최적 가중치는 `models/fhe_cnn_fer2013_2.pt`, 입력 정규화 통계는 `models/normalization_stats.json`으로 저장됩니다.

## 암호화 추론 데모
- 단일 샘플에 대해 평문/암호문 추론 결과를 비교합니다.
- CLI 실행:
  ```bash
  cd fhe_emotion
  ./scripts/run_encrypted_infer.sh
  ```
- 노트북:
  - `notebooks/03_tenseal_encrypted_inference.ipynb`: 단일 이미지에 대한 평문/암호문 추론 비교 및 시각화
  - `notebooks/04_tenseal_test_accuracy.ipynb`: 다중 샘플(20개)에 대한 정확도 테스트 및 추론 시간 측정
- TenSEAL 추론 실행기:
  - `PackedEncryptedCNNRunner`(기본): im2col 인코딩으로 입력을 단일 CKKSVector에 패킹하고 `conv2d_im2col` 연산을 사용합니다. 추론 시간 약 18-20초/샘플.
  - `EncryptedCNNRunner`: 픽셀별 스칼라 암호문을 사용하는 디버그용 구현입니다. `encrypted_inference_demo(..., use_packed=False)`로 호출할 수 있습니다.
- **CKKS 컨텍스트 파라미터** (`he/tenseal_context.py`):
  - `poly_modulus_degree`: 32768 (16,384 슬롯 제공)
  - `coeff_mod_bit_sizes`: [31, 26, 26, 26, 26, 26, 26, 31]
  - `global_scale`: 2^26
  - Multiplicative depth 지원: 3-4 레벨 (Conv→Square→FC→Square)

## 디렉터리 및 파일 설명
- `fhe_emotion/models/fhe_cnn.py`: Wide 1-Conv 아키텍처 정의 (16 channels, Stride 3, Square activation, no BN)
  - 구조: Conv(1→16, k=7, s=3) → Square → Flatten(3136) → FC(3136→128) → Square → FC(128→7)
  - TenSEAL용 파라미터 추출 도우미 포함
- `fhe_emotion/he/tenseal_context.py`: CKKS 컨텍스트 생성 (poly_modulus=32768), 벡터 암복호화, 직렬화 유틸리티
- `fhe_emotion/he/fhe_inference.py`: PyTorch 가중치 로딩, im2col 기반 패킹 추론, `encrypted_inference_demo` 제공
- `fhe_emotion/notebooks/`:
  - `01_prepare_fer2013.ipynb`: FER2013 데이터셋 전처리
  - `02_train_plain_cnn.ipynb`: 평문 CNN 학습 (50 epochs)
  - `03_tenseal_encrypted_inference.ipynb`: 단일 샘플 암호화 추론 데모
  - `04_tenseal_test_accuracy.ipynb`: 다중 샘플 정확도 테스트
- `fhe_emotion/scripts/`:
  - `setup_env.sh`: 프로젝트 의존성 설치 및 Jupyter 커널 등록
  - `run_train.sh`: 학습 노트북 비대화식 실행
  - `run_encrypted_infer.sh`: TenSEAL 추론 모듈 CLI 실행
- `fhe_emotion/data/processed/`: 전처리 후 저장되는 텐서 및 클래스 가중치

## 참고자료
- 데이터셋: [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- FHE 패턴: [TenSEAL Tutorial 4 - Encrypted Convolution on MNIST](https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb)

## 주요 설계 결정
- **Batch Normalization 제거**: FHE 추론 단순화를 위해 BN 완전 제거
- **Stride 3 사용**: Stride 2(21×21=441 windows) 대신 Stride 3(14×14=196 windows)으로 CKKS 슬롯 요구량 감소
- **16 채널 사용**: 16×196=3,136 슬롯으로 32768 poly_modulus_degree에 여유롭게 수용
- **Multiplicative Depth 최소화**: Conv→Square→FC→Square 총 4단계로 CKKS 노이즈 누적 최소화

필요한 부분(학습 epoch 수, 컨텍스트 파라미터 등)을 조정하면 더 빠른 실험이나 정확도 개선을 쉽게 시도할 수 있습니다.
