/video_transcript_project/
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
# ctranslate2 강제 재설치 (이전 단계에서 했던 것처럼) 4.6 버젼 설치되어야 torch gpu 사용가능
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade --force-reinstall "ctranslate2[cuda]"

# CUDNN 충돌로 안될 경우
1단계: 완전 초기화
1. .venv폴더를 완전히 삭제합니다.
2. 명령 프롬프트에서 아래 명령어로 새로운 가상 환경을 만들고 활성화합니다.
2단계: 라이브러리 설치 (순서 및 방법이 매우 중요!)
1. pip 업그레이드
   python -m pip install --upgrade pip
2. whisperx먼저 설치하기 (의존성 포함)
   ctranslate2를 먼저 설치하는 것이 아니라, whisperx를 먼저 설치해서 필요한 모든 의존성 패키지(구버전 ctranslate2포함)를 한번에 설치하게 둡니다.
   pip install git+https://github.com/m-bain/whisperX.git
   이 단계가 끝나면, 시스템에는 ctranslate2-4.4.0이 설치된 상태가 됩니다.
3. torch설치 (GPU 지원)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
4. ctranslate2강제 업그레이드 (핵심 단계)
   이제, whisperx가 설치해놓은 구버전 ctranslate2를 우리가 원하는 최신 CUDA 지원 버전으로 강제로 덮어씌웁니다.
   pip install --upgrade --force-reinstall "ctranslate2[cuda]"
	- --upgrade: 최신 버전으로 설치
	- --force-reinstall: 이미 설치되어 있더라도 강제로 재설치. 이것이 whisperx의 의존성 제약을 무시하고 우리가 원하는 버전을 설치하게 만드는 핵심 옵션입니다.
5. 나머지 라이브러리 설치
   pip install ffmpeg-python python-dotenv
3단계: 최종 확인 및 실행
1. 설치가 끝난 후, pip list명령어로 설치된 패키지 목록을 확인해보세요. ctranslate2의 버전이 4.6.0으로, torch가 2.5.1+cu121로 되어 있으면 성공입니다.


# API 
호출 : http://127.0.0.1:5001/speaker?path=D:\test.mp4&key=11111&model=large-v3
리턴 : http://127.0.0.1/speaker_sucess.php/speaker_sucess.php?key=11111&path=D%3A%5Ctest_speaker.txt  