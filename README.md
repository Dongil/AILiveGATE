# 프로젝트 구성
/project/

├── .venv/                     # 파이썬 가상 환경

├── main.py                    # FastAPI 서버 실행 파일

├── requirements.txt           # 프로젝트 의존성 목록

├── config.py                  # 설정 파일 (콜백 URL, 경로 등)

└── processor/

    ├── __init__.py

    └── tasks.py               # 실제 회의록 생성 로직이 담긴 파일

# 의존성 설치
pip install -r requirements.txt

pip install git+https://github.com/m-bain/whisperX.git

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDNN 충돌로 안될 경우
1단계: 완전 초기화
1. .venv폴더를 완전히 삭제합니다.
2. 명령 프롬프트에서 아래 명령어로 새로운 가상 환경을 만들고 활성화합니다.
2단계: 라이브러리 설치 (순서 및 방법이 매우 중요!)
1. pip 업그레이드
   python -m pip install --upgrade pip
2. whisperx먼저 설치하기 (의존성 포함)
   pip install git+https://github.com/m-bain/whisperX.git 
3. torch설치 (GPU 지원)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# API 명세 

## 1. 화자 분리  
호출 : http://127.0.0.1:5001/speaker?path=D:\test.mp4&key=11111&model=large-v3&threshold=0.7&min_duration_off=0.2&min_speakers=10&max_speakers=15  
(튜닝 파라메타 threshold, min_duration_off, min_speakers, max_speakers 는 옵션) (기본값 : threshold=0.7, min_duration_off=0.2, min_speakers=2, max_speakers=25)  
리턴 : http://127.0.0.1/speaker_sucess.php?key=11111&path=D:\test.txt  
      -d:\test.vtt 동시 생성 (<speaker_00> 추가된 WebVTT파일)

      * Diarze 파라메타 튜닝
      각 하이퍼파라미터의 의미와 튜닝 전략
      - threshold (가장 중요)
      의미: 클러스터링 임계값입니다. 두 음성 발화의 임베딩(음색 특징 벡터) 사이의 거리를 계산하여, 이 값이 threshold보다 작으면 "같은 사람", 크면 "다른 사람"으로 판단합니다. 값의 범위는 보통 0에서 2 사이입니다.
      튜닝 전략:
      현재 문제: "사람이 바뀌었는데도 중복으로 표시하는 경향" → 여러 사람을 한 사람으로 너무 관대하게 묶고 있다는 의미.
      해결책: threshold 값을 낮춰야 합니다. 기준을 더 엄격하게 만들어, 약간의 음색 차이만 있어도 다른 사람으로 인식하도록 유도합니다.
      추천 값: 기본값은 모델마다 다르지만 보통 0.5 ~ 0.8 사이입니다. 0.7에서 시작하여 0.6, 0.5 등으로 점차 낮춰보면서 결과가 어떻게 변하는지 확인하세요. 너무 낮추면 반대로 한 사람이 말하는 중간에 화자가 바뀌는 문제가 발생할 수 있습니다.
      
      - min_duration_off
      의미: 음성 활동 감지(VAD) 후, 음성이 없는 구간(silence)의 최소 지속 시간입니다. 이 값보다 짧은 침묵은 무시하고 앞뒤 발언을 하나의 세그먼트로 간주합니다.
      튜닝 전략:
      현재 문제: 지방 의회 회의는 한 사람이 길게 발언하는 도중에 짧은 쉼(pause)을 갖는 경우가 많습니다.
      min_duration_off 값을 너무 낮게 잡으면 (예: 0.1): 짧은 쉼마다 발언이 끊어져, 한 사람의 발언이 여러 조각으로 나뉘고, 이 조각들이 다른 화자로 오인될 수 있습니다.
      min_duration_off 값을 적절히 높이면 (예: 0.5 ~ 1.0): 발언 중간의 어지간한 쉼은 무시하고 하나의 긴 발언으로 처리하여 화자 분리의 안정성을 높일 수 있습니다.
      추천 값: 기본값(보통 0.2 근처)에서 시작하여, 한 사람의 발언이 너무 잘게 쪼개지는 것 같다면 0.5 나 0.8 로 올려서 테스트해보세요.
      
      추천 튜닝 순서
      1단계: threshold 값 조정 (가장 먼저)
      여러 사람을 하나로 묶는 문제를 해결하기 위해 0.7 -> 0.6 -> 0.5 순으로 낮춰보며 테스트합니다.
      2단계: min_duration_off 값 조정
      한 사람의 발언이 불필요하게 쪼개지는지 확인하고, 그렇다면 0.2 -> 0.5 -> 0.8 순으로 높여보며 테스트합니다.
      3단계: min_speakers, max_speakers와 조합
      위의 파라미터들과 함께, 회의에 참석한 대략적인 인원수를 min/max_speakers에 지정해주면 모델이 탐색할 공간을 제한하여 정확도를 높일 수 있습니다. (pyannote 파이프라인도 이 인자를 받습니다.)
      이러한 하이퍼파라미터 튜닝은 정답이 없으며, 대상 오디오의 특성(화자 수, 대화 속도, 배경 소음 등)에 따라 최적값이 달라집니다. 몇 번의 실험을 통해 "우리 회의 영상에는 이 조합이 가장 좋구나"하는 최적의 값을 찾아가는 과정이 필요합니다.

## 2. MP4 -> mp3, wav 오디오 파일 생성  
호출 : http://127.0.0.1:5001/audio_convert?path=D:\test.mp4&key=1111&type=wav  
리턴 : http://127.0.0.1/audio_sucess.php?key=11111&path=D%3A%5Ctest.txt&type=wav  
      -Type : mp3, wav  (Clova Speech API 연동 적합한 mp3 코덱, google STT 연동 적합한 wav 코덱)

       