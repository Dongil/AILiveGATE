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

# API 
1. 화자 분리  
호출 : http://127.0.0.1:5001/speaker?path=D:\test.mp4&key=11111&model=large-v3  
리턴 : http://127.0.0.1/speaker_sucess.php?key=11111&path=D%3A%5Ctest.txt  
      -d:\test.vtt 동시 생성 (<speaker_00> 추가된 WebVTT파일)

2. MP4 -> mp3, wav 오디오 파일 생성  
호출 : http://127.0.0.1:5001/audio_convert?path=D:\test.mp4&key=1111&type=wav  
리턴 : http://127.0.0.1/audio_sucess.php?key=11111&path=D%3A%5Ctest.txt&type=wav  
      -Type : mp3, wav  (Clova Speech API 연동 적합한 mp3 코덱, google STT 연동 적합한 wav 코덱)