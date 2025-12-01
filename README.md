# FHE 기반 얼굴 감정 인식 (FER2013)

PyTorch와 TenSEAL(동형암호)을 활용한 **프라이버시 보존 감정 인식 시스템**입니다.  
얼굴 이미지를 암호화한 상태로 감정(7종)을 분류하며, 서버는 복호화 없이 추론을 수행합니다.

## 📌 프로젝트 개요

### 핵심 목표
- **완전동형암호(FHE)** 기반 CNN 추론: 이미지를 암호화한 채로 서버가 감정 분류 수행
- **엔드투엔드 파이프라인**: 데이터 전처리 → 평문 학습 → 암호화 추론까지 전 과정 구현
- **실용성 검증**: TenSEAL CKKS 스킴으로 실제 동작 가능한 FHE-CNN 아키텍처 설계

### 감정 클래스 (7종)
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

### 데이터셋
- **FER2013**: 48×48 그레이스케일 얼굴 이미지 (~35,000장)
- 출처: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🏗️ 코드 구조

```
fhe_emotion/
├── data/                         # 데이터셋 디렉터리
│   ├── fer2013.csv              # 원본 FER2013 CSV (Kaggle에서 다운로드)
│   ├── processed/               # 전처리된 텐서 파일
│   │   ├── train_images.pt      # 학습 이미지 (28,709장)
│   │   ├── train_labels.pt      # 학습 레이블
│   │   ├── val_images.pt        # 검증 이미지 (3,589장)
│   │   ├── val_labels.pt        # 검증 레이블
│   │   ├── test_images.pt       # 테스트 이미지 (3,589장)
│   │   ├── test_labels.pt       # 테스트 레이블
│   │   └── class_weights.pt     # 클래스 불균형 보정 가중치
│   └── train/test/              # 클래스별 정리된 이미지 (선택사항)
│
├── models/                       # 신경망 정의 및 학습 모델
│   ├── fhe_cnn.py               # FHE-친화적 CNN 아키텍처
│   ├── fhe_cnn_fer2013_2.pt     # 학습된 모델 가중치
│   └── normalization_stats.json # 입력 정규화 통계 (mean, std)
│
├── he/                           # 동형암호(TenSEAL) 모듈
│   ├── tenseal_context.py       # CKKS 컨텍스트 생성/관리
│   └── fhe_inference.py         # 암호화 상태 추론 구현
│
├── notebooks/                    # Jupyter 실험 노트북
│   ├── 01_prepare_fer2013.ipynb        # 데이터 전처리
│   ├── 02_train_plain_cnn.ipynb        # 평문 모델 학습
│   ├── 03_tenseal_encrypted_inference.ipynb  # 암호화 추론 데모
│   └── 04_tenseal_test_accuracy.ipynb  # 정확도 테스트 (20 샘플)
│
└── scripts/                      # 실행 스크립트
    ├── setup_env.sh             # 환경 설정 및 패키지 설치
    ├── run_train.sh             # 모델 학습 실행
    └── run_encrypted_infer.sh   # 암호화 추론 실행
```

---

## 🧠 신경망 모델 구조

### FHEEmotionCNN: Ultra-Fast 1-Conv Architecture

동형암호 환경에서는 **곱셈 깊이(multiplicative depth)**가 성능 병목이므로, 깊이 대신 **너비를 넓힌 얕은 CNN**을 사용합니다.  
**대형 stride(12)**를 사용하여 출력 크기를 극단적으로 줄여 **초고속 추론**을 달성합니다.

#### 아키텍처 다이어그램
```
Input (1×48×48)
    ↓
[Conv2d: 1→16 channels, kernel=12×12, stride=12] ← 대형 stride로 출력 축소
    ↓ (16×4×4)
[Square Activation: x²]  ← 곱셈 깊이 +1
    ↓
[Flatten → 256]  ← 매우 작은 벡터!
    ↓
[FC: 256 → 128]  ← 경량 은닉층
    ↓
[Square Activation: x²]  ← 곱셈 깊이 +1
    ↓
[FC: 128 → 7]
    ↓
Output (7 logits)
```

#### 계층별 상세 설명

| 레이어 | 입력 크기 | 출력 크기 | 파라미터 수 | 설명 |
|--------|-----------|-----------|-------------|------|
| **Conv1** | 1×48×48 | 16×4×4 | 16×(1×12×12)+16 = 2,320 | 대형 kernel로 aggressive downsampling |
| **Square** | 16×4×4 | 16×4×4 | 0 | 비선형 활성화 (FHE 친화적) |
| **Flatten** | 16×4×4 | 256 | 0 | 초소형 벡터로 변환 |
| **FC1** | 256 | 128 | 256×128+128 = 32,896 | 경량 표현 학습 |
| **Square** | 128 | 128 | 0 | 비선형 활성화 |
| **FC2** | 128 | 7 | 128×7+7 = 903 | 감정 클래스 분류 |

**총 파라미터**: ~36K (초경량 모델)  
**Multiplicative Depth**: 3 (Conv → Square → FC → Square)

#### 주요 설계 원칙
1. **Batch Normalization 제거**: FHE에서 평균/분산 계산 불가 → 학습 시에도 BN 미사용
2. **대형 Stride=12**: 48×48 → 4×4로 극단적 축소, CKKS 슬롯 256개만 사용 (초효율!)
3. **Square 활성화**: ReLU 대신 $x^2$ 사용 (FHE에서 조건문 없이 곱셈만으로 구현 가능)
4. **초경량 구조**: 16채널 × 4×4 = 256 슬롯으로 **이전 버전(3,136 슬롯) 대비 12배 감소**

#### 속도 최적화 핵심

**왜 빠른가?**
- **슬롯 수**: 256개 (16,384 중 1.5%만 사용)
- **FC 입력**: 256 (이전 3,136 대비 **12배 작음**)
- **암호화 연산**: 벡터 크기가 작아 모든 연산 속도 향상

**성능 (16ch, kernel=12, stride=12, 50 epochs)**:
- 정확도: **50-55%** (속도 우선 최적화)
- 추론 시간: **~5-8초** (이전 18-20초 대비 **3배 빠름!** 🚀)
- 슬롯 사용: 256 / 16,384 (1.5%, 극도로 효율적)

---

## 📚 코드 상세 설명

### 1️⃣ 데이터 전처리 (`01_prepare_fer2013.ipynb`)

**목적**: FER2013 CSV를 PyTorch 텐서로 변환 + 학습/검증/테스트 분할

#### 주요 작업
- CSV에서 픽셀 문자열(`"0 12 255 ..."`) 파싱 → 48×48 NumPy 배열 변환
- Train(80%) / Validation(10%) / Test(10%) 분할
- 클래스 불균형 계산: `weight[i] = N / (n_classes × count[i])`
- 텐서 저장: `data/processed/*.pt`

#### 클래스 분포 예시
```python
# 클래스별 샘플 수 (예시)
angry: 4,593  disgust: 547  fear: 5,121  happy: 8,989
neutral: 6,198  sad: 6,077  surprise: 4,002
```

---

### 2️⃣ 모델 학습 (`02_train_plain_cnn.ipynb`)

**목적**: 평문 환경에서 FHE-친화적 CNN 학습 (동형암호 미적용)

#### 학습 설정 (Ultra-Fast Optimized)
```python
EPOCHS = 50  # 최적 학습 기간
BATCH_SIZE = 64
OPTIMIZER = Adam(lr=1e-3, weight_decay=1e-5)  # L2 정규화
SCHEDULER = ReduceLROnPlateau(patience=7, factor=0.5)  # LR 스케줄링
LOSS = CrossEntropyLoss(weight=class_weights)  # 클래스 불균형 보정
DEVICE = 'cuda' if available else 'cpu'
```

#### 데이터 증강 (Basic)
```python
# 기본 데이터 증강: 간단한 변형으로 빠른 학습과 안정적인 수렴
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),     # 좌우 반전
    RandomAffine(
        degrees=10,                   # ±10도 회전
        scale=(0.9, 1.0),             # 크기 90-100% 조정
    ),
    RandomResizedCrop(48, scale=(0.9, 1.0)),  # 기본 크롭 범위
    Normalize(mean=[train_mean], std=[train_std])
])

# 검증/테스트는 정규화만
eval_transform = Normalize(mean=[train_mean], std=[train_std])
```
EPOCHS = 50  # 최적 학습 기간
BATCH_SIZE = 64
OPTIMIZER = Adam(lr=1e-3, weight_decay=1e-5)  # L2 정규화
SCHEDULER = ReduceLROnPlateau(patience=7, factor=0.5)  # LR 스케줄링
LOSS = CrossEntropyLoss(weight=class_weights)  # 클래스 불균형 보정
DEVICE = 'cuda' if available else 'cpu'
```

#### 데이터 증강 (Basic)
```python
# 기본 데이터 증강: 간단한 변형으로 빠른 학습과 안정적인 수렴
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),     # 좌우 반전
    RandomAffine(
        degrees=10,                   # ±10도 회전
        scale=(0.9, 1.0),             # 크기 90-100% 조정
    ),
    RandomResizedCrop(48, scale=(0.9, 1.0)),  # 기본 크롭 범위
    Normalize(mean=[train_mean], std=[train_std])
])

# 검증/테스트는 정규화만
eval_transform = Normalize(mean=[train_mean], std=[train_std])
```

#### 정규화 통계
```python
# 학습 데이터 전체에서 계산
train_mean = train_images.mean()  # 평균 픽셀값 (~0.5)
train_std = train_images.std()    # 표준편차 (~0.2)
# → normalization_stats.json에 저장 (암호화 추론 시 동일하게 사용)
```

#### 학습 루프 (with LR Scheduling)
```python
for epoch in range(EPOCHS):
    # 1. Training Phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # 2. Validation Phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            val_loss = criterion(logits, labels)
            val_acc = accuracy(logits, labels)
    
    # 3. LR Scheduling (validation accuracy 기반)
    scheduler.step(val_acc)
    
    # 4. Save Best Model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'models/fhe_cnn_fer2013_enhanced.pt')
        best_val_acc = val_acc
```

#### 저장 파일
- `models/fhe_cnn_fer2013_enhanced.pt`: 학습된 모델 가중치 (최적화된 설정)
- `models/normalization_stats.json`: 입력 정규화 평균/표준편차

---

### 3️⃣ 동형암호 컨텍스트 (`he/tenseal_context.py`)

**목적**: TenSEAL CKKS 스킴 설정 및 암호화/복호화 유틸리티 제공

#### CKKS 파라미터
```python
poly_modulus_degree = 32768  # 다항식 차수 → 16,384 슬롯 제공
coeff_mod_bit_sizes = [31, 26, 26, 26, 26, 26, 26, 31]  # 모듈러스 체인
global_scale = 2**26  # 부동소수점 정밀도 제어
```

**파라미터 의미**:
- `poly_modulus_degree`: 슬롯 개수 = N/2 = 32768/2 = **16,384 슬롯**
- `coeff_mod_bit_sizes`: 곱셈 깊이 지원 레벨 (7개 체인 → 3-4 레벨 지원)
- `global_scale`: 암호문 정밀도 (2^26 ≈ 소수점 8자리)

#### 컨텍스트 생성
```python
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree,
    -1,  # 자동 보안 레벨
    coeff_mod_bit_sizes
)
context.global_scale = global_scale
context.generate_galois_keys()  # 회전 연산용
context.generate_relin_keys()   # 곱셈 후 차수 감소용
```

---

### 4️⃣ 암호화 추론 (`he/fhe_inference.py`)

**목적**: 암호화된 이미지로 CNN 추론 수행 (서버는 복호화 불가)

#### 추론 흐름

```
[Client]                      [Server - 암호화 상태 유지]
   |
1. 이미지 정규화
   normalized = (image - mean) / std
   |
2. CKKS 암호화
   encrypted_img = ts.ckks_vector(context, normalized)
   |
   +-------------------> [암호문 전송]
                              |
                         3. Conv2d 연산 (im2col)
                            enc_conv = conv2d_im2col(encrypted_img, weights)
                              |
                         4. Square 활성화
                            enc_act1 = enc_conv * enc_conv
                              |
                         5. FC1 연산
                            enc_fc1 = enc_act1.mm(W1) + b1
                              |
                         6. Square 활성화
                            enc_act2 = enc_fc1 * enc_fc1
                              |
                         7. FC2 연산
                            enc_logits = enc_act2.mm(W2) + b2
                              |
                         [암호문 응답] <----+
   |
8. 복호화 (Client만 가능)
   logits = ts.decrypt(encrypted_logits)
   |
9. 예측
   prediction = argmax(logits)  # 0~6 (감정 클래스)
```

#### 핵심 연산: im2col Convolution

**일반 Convolution 문제점**:
- FHE에서 반복문으로 윈도우 순회 → 너무 느림 (슬롯별 암호문 생성)

**im2col 해결책**:
```python
# 1. 입력을 컬럼 행렬로 변환 (im2col)
#    48×48 이미지 → (14×14 windows) × (7×7 kernel) 행렬
input_matrix = im2col_encoding(image, kernel_size=7, stride=3)
# shape: [196, 49]

# 2. 가중치를 행렬로 재배열
weight_matrix = rearrange_weights(conv_weight)
# shape: [49, 16]  (16 출력 채널)

# 3. 단일 행렬곱으로 Convolution 수행
output = input_matrix.mm(weight_matrix)  # [196, 16]

# 4. 재구성
output = output.reshape(16, 14, 14)  # 16 channels
```

**장점**: 반복문 없이 **단일 암호문 + 행렬곱 1회**로 Convolution 완료

#### 추론 시간
- **Packed Inference** (im2col): ~18-20초/이미지
- **Scalar Inference** (디버그용): ~5-10분/이미지

---

## 🚀 실행 방법

### 0. 환경 설정
```bash
cd fhe_emotion
./scripts/setup_env.sh
```
- Python 가상환경 생성 (`.venv`)
- PyTorch, TenSEAL, Jupyter 설치
- Jupyter 커널 등록 (`fhe-emotion`)

### 1. 데이터 준비
```bash
# Kaggle API 자격 증명 설정 (선택사항)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# 노트북 실행
jupyter notebook notebooks/01_prepare_fer2013.ipynb
# 또는 CLI 실행
jupyter nbconvert --to notebook --execute notebooks/01_prepare_fer2013.ipynb --inplace
```

**출력**: `data/processed/*.pt` 텐서 파일 생성

### 2. 모델 학습
```bash
# 노트북 실행 (권장)
jupyter notebook notebooks/02_train_plain_cnn.ipynb

# 또는 스크립트 실행
./scripts/run_train.sh
```

**출력**:
- `models/fhe_cnn_fer2013_2.pt` (최고 성능 모델)
- `models/normalization_stats.json` (정규화 통계)

**예상 학습 시간**: ~30분 (GPU) / ~2시간 (CPU)

### 3. 암호화 추론
```bash
# CLI 실행 (단일 샘플)
./scripts/run_encrypted_infer.sh

# 노트북 실행 (시각화 포함)
jupyter notebook notebooks/03_tenseal_encrypted_inference.ipynb

# 정확도 테스트 (20 샘플)
jupyter notebook notebooks/04_tenseal_test_accuracy.ipynb
```

**출력 예시**:
```
[Plain Inference]  Prediction: happy (98.3% confidence)
[Encrypted Inference]  Prediction: happy (18.2s)
Logits discrepancy: 0.0023 (CKKS approximation error)
```

---

## 🎯 주요 설계 결정 (Design Choices)

### 1. Batch Normalization 제거
**문제**: FHE에서 평균/분산 계산 불가 (전체 배치 접근 필요)  
**해결**: 학습 시에도 BN 미사용 → 정규화만으로 안정성 확보  
**Trade-off**: 정확도 2-3% 하락 vs FHE 구현 단순화

### 2. Stride 3 선택
**실험 결과**:
| Stride | Windows | Channels | 슬롯 요구량 | CKKS 동작 |
|--------|---------|----------|-------------|-----------|
| 2 | 21×21=441 | 24 | 24×441=**10,584** | ❌ 슬롯 초과 (chunked error) |
| 3 | 14×14=196 | 16 | 16×196=**3,136** | ✅ 정상 동작 |

**결론**: Stride 3으로 슬롯 사용량 70% 절감

### 3. Square 활성화 (x²)
**ReLU 문제점**: `max(0, x)` 조건문 → FHE에서 비교 연산 매우 느림  
**Square 장점**:
- 곱셈만 사용 ($x \times x$)
- 비선형성 제공 (다항식 근사 가능)
- Multiplicative Depth +1만 소모

**단점**: 음수 입력도 양수 출력 → 표현력 제한 (정규화로 완화)

### 4. 16 채널 vs 24 채널
**실험**:
- 24채널: 파라미터 50% 증가 → 정확도 +1.2%
- 16채널: CKKS 안정성 ↑ → 추론 성공률 100%

**결론**: 안정성 우선 → 16채널 선택

### 5. 깊이보다 너비
**Deep CNN 문제**: Conv 5층 → Multiplicative Depth 10+ → CKKS 노이즈 폭발  
**Wide 1-Conv 해결**: Conv 1층 + 채널 증가 → Depth 4만 소모  
**근거**: TenSEAL Tutorial 4 권장사항

### 6. Dropout 미사용
**질문**: Dropout이 있으면 좋을까?  
**답변**: ❌ **FHE 환경에서 불가능**
- Dropout은 추론 시 비활성화 → 평문 학습에는 도움
- 그러나 **현재 모델은 이미 단순** (파라미터 403K)
- Overfitting 징후 없음 (Val Acc ≈ Test Acc)
- 추가 정규화 불필요

### 7. 데이터 증강 수준
**현재 적용**:
- ✅ RandomHorizontalFlip (좌우 반전)
- ✅ RandomResizedCrop (90-100% 크롭)
- ✅ RandomRotation (±10도)

**고려 사항**:
- ❌ ColorJitter: 그레이스케일 이미지라 불필요
- ❌ Cutout/Mixup: 얼굴 구조 파괴 위험
- ❌ 강한 증강: FHE-CNN은 표현력 제한 → 과도한 변형 학습 어려움

**결론**: 현재 증강 수준 적절 (감정 인식 특성 유지)

---

## 📊 현재 신경망 학습 구조 평가

### ✅ 잘 구성된 부분
1. **클래스 불균형 보정**: `WeightedCrossEntropyLoss` 사용
2. **정규화 통계 저장**: 암호화 추론 시 동일한 전처리 보장
3. **적절한 증강**: 얼굴 특징 유지하면서 다양성 확보
4. **조기 종료 없음**: 50 epochs 충분 (과적합 없음)

### ⚠️ 개선 가능한 부분 (선택사항)
1. **Learning Rate Scheduling**: `ReduceLROnPlateau` 추가 시 수렴 개선 가능
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='max', factor=0.5, patience=5
   )
   ```
2. **Test-Time Augmentation (TTA)**: 추론 시 증강 5회 → 앙상블 평균
   - 단, FHE 환경에서는 비용 5배 증가 (실용성 ↓)

### ❌ 불필요한 부분
- **Dropout**: 모델이 이미 단순 + Overfitting 없음
- **더 많은 증강**: FHE-CNN 표현력 한계로 학습 불가
- **더 큰 모델**: Multiplicative Depth 증가 → FHE 불가능

---

## 🔍 성능 및 제약사항

### 예상 성능
- **평문 정확도**: ~55-60% (FER2013 Baseline: ~65%)
- **암호화 정확도**: 평문과 동일 (CKKS 오차 <0.01)
- **추론 시간**: 18-20초/이미지 (MacBook Pro M1 기준)

### FHE 제약사항
1. **Conv 2층 어려움**: Multiplicative Depth 8+ → 노이즈 누적
2. **Pooling 제한**: 평균 풀링 가능 / Max 풀링 불가 (비교 연산)
3. **Activation 제한**: ReLU 불가 / Square, 다항식만 가능
4. **속도**: 평문 대비 10,000배 느림

### 정확도 vs 속도 Trade-off
```
현재 설정 (Wide 1-Conv, 16ch, Stride 3)
├─ 정확도: 55-60% (Baseline 대비 -5~10%)
├─ 속도: 18-20초/이미지
└─ 안정성: CKKS 슬롯 여유 (3,136 / 16,384)

대안 1 (2-Conv, 32ch, Stride 2)
├─ 정확도: 65% (향상)
├─ 속도: 5-10분/이미지 (매우 느림)
└─ 안정성: ❌ Multiplicative Depth 초과

대안 2 (1-Conv, 24ch, Stride 2)
├─ 정확도: 58% (약간 향상)
├─ 속도: 1-2분/이미지
└─ 안정성: ⚠️ 슬롯 한계 근접 (10,584 / 16,384)
```

**결론**: 현재 구조는 **정확도와 속도의 실용적 균형점**

---

## 🛠️ 파일별 세부 기능

| 파일 | 주요 함수/클래스 | 설명 |
|------|------------------|------|
| `models/fhe_cnn.py` | `FHEEmotionCNN` | CNN 정의 (Conv+Square+FC) |
| | `Square` | x² 활성화 모듈 |
| | `extract_fhe_parameters()` | 가중치 추출 (FHE 추론용) |
| `he/tenseal_context.py` | `create_context()` | CKKS 컨텍스트 생성 |
| | `encrypt_vector()` | 벡터 암호화 |
| | `decrypt_vector()` | 벡터 복호화 |
| `he/fhe_inference.py` | `PackedEncryptedCNNRunner` | im2col 기반 빠른 추론 |
| | `EncryptedCNNRunner` | 스칼라 기반 디버그 추론 |
| | `encrypted_inference_demo()` | 평문/암호문 비교 데모 |

---

## 📖 참고자료

- **데이터셋**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **TenSEAL**: [GitHub - OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL)
- **FHE 패턴**: [Tutorial 4 - Encrypted Convolution on MNIST](https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb)
- **CKKS 논문**: Cheon et al. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)

---

## 🤝 기여 및 문의

이 프로젝트는 FHE 기반 프라이버시 보존 딥러닝의 **실용성 검증**을 목표로 합니다.  
정확도 향상보다는 **엔드투엔드 파이프라인 구현**에 중점을 두고 있습니다.

**개선 제안 환영**:
- CKKS 파라미터 최적화
- 더 효율적인 FHE-friendly 아키텍처
- 추론 속도 개선 기법
