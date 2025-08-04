# /config.py
import os
from dotenv import load_dotenv

load_dotenv() # API key가져옴

HF_TOKEN = os.getenv("HF_TOKEN")

# -- 콜백(Callback) 설정 --
# 작업 완료 후 호출할 CMS의 기본 URL
SPEAKER_CALLBACK_URL = "http://127.0.0.1/speaker_sucess.php"
AUDIO_CALLBACK_URL = "http://127.0.0.1/audio_sucess.php" # 새로 추가

# -- 모델 기본 설정 --
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# -- 후처리 설정 --
MERGE_THRESHOLD_SECONDS = 2.0 
SHORT_SEGMENT_WORD_COUNT = 3