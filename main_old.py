# main.py

import whisperx
import gc
import ffmpeg # ffmpeg-python 라이브러리
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

load_dotenv()

# -----------------
# 설정
# -----------------
#video_path = r"D:\성동구의회 본회의.mp4"
# video_path = r"D:\250213 제229회 안성시의회 임시회 제1차 업무계획청취특별위원회 (2).mp4"
video_path = r"D:\test.mp4"
audio_path = r"D:\temp_audio.wav"
device = "cuda" # or "cpu", "cuda"
batch_size = 16
beam_size = 10
# CPU 사용 시 float32가 더 안정적일 수 있습니다.
compute_type = "float32" if device == "cpu" else "float16"

# -----------------
# 1. MP4에서 오디오 추출
# -----------------
print("1. 오디오 추출 중...")
try:
    # ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)
    print(f"오디오 추출 완료: {audio_path}")
except ffmpeg.Error as e:
    print('ffmpeg 오류가 발생했습니다.')
    if e.stderr:
        print('ffmpeg stderr:', e.stderr.decode('utf8', errors='ignore'))
    exit()

# -----------------
# 2. WhisperX를 이용한 화자 분리 및 ASR
# -----------------
print("2. WhisperX 모델 로드 중...")
# whisper 모델 로드
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# 오디오 파일 로드 및 whisper 실행
print("   - 음성 인식(ASR) 진행 중...")
audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio, batch_size=batch_size)

# 메모리 정리
del model
gc.collect()

# 타임스탬프 정렬 모델 로드 및 실행
print("   - 타임스탬프 정렬 중...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 메모리 정리
del model_a, metadata
gc.collect()

# 화자 분리 모델 로드 및 실행
# .env 파일에 HF_TOKEN="hf_..."를 설정하거나, huggingface-cli login을 미리 실행해야 함
print("   - 화자 분리 진행 중...")
diarize_model = DiarizationPipeline(device=device)
diarize_segments = diarize_model(audio) # 화자 수 힌트 제공 가능

# 결과 병합
print("   - 결과 병합 중...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# -----------------
# 3. 결과 후처리 및 출력 (향상된 버전)
# -----------------

# -- 후처리 설정값 --
# 이 시간(초)보다 짧은 간격으로 이어진 발언은 같은 화자로 간주하여 병합 시도
MERGE_THRESHOLD_SECONDS = 2.0 
# 이 단어 수 이하의 짧은 발언을 병합 대상으로 간주
SHORT_SEGMENT_WORD_COUNT = 3

print("\n--- 최종 회의록 (후처리 적용) ---")

# 1. 초기 세그먼트 정리 및 형식 변환
processed_segments = []
if "segments" in result and result["segments"]:
    for seg in result["segments"]:
        # 'speaker'가 없는 경우 'UNKNOWN'으로 처리
        if 'speaker' not in seg:
            seg['speaker'] = 'UNKNOWN'
        
        # 각 세그먼트의 시작/종료 시간을 hh:mm:ss 형식으로 변환
        start_time = int(seg['start'])
        end_time = int(seg['end'])
        
        # 타임스탬프 포맷팅 함수
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        processed_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'timestamp': format_time(start_time),
            'speaker': seg['speaker'],
            'text': seg['text'].strip()
        })

# 2. 짧은 발언 병합 로직
merged_segments = []
if processed_segments:
    # 첫 번째 세그먼트는 일단 추가
    merged_segments.append(processed_segments[0])

    for i in range(1, len(processed_segments)):
        current_seg = processed_segments[i]
        prev_seg = merged_segments[-1] # 병합된 리스트의 마지막 세그먼트

        # 현재 세그먼트의 단어 수 계산
        current_word_count = len(current_seg['text'].split())
        
        # 병합 조건 확인:
        # 1. 이전 발언이 끝난 후 MERGE_THRESHOLD_SECONDS 안에 현재 발언이 시작되었는가?
        # 2. 현재 발언의 단어 수가 SHORT_SEGMENT_WORD_COUNT 이하인가?
        # 3. (선택적) 또는 화자가 UNKNOWN 인가?
        time_gap = current_seg['start'] - prev_seg['end']
        
        if (time_gap < MERGE_THRESHOLD_SECONDS and current_word_count <= SHORT_SEGMENT_WORD_COUNT) or current_seg['speaker'] == 'UNKNOWN':
            # -- 병합 수행 --
            # 텍스트를 이전 세그먼트에 합치기
            prev_seg['text'] += " " + current_seg['text']
            # 종료 시간을 현재 세그먼트의 종료 시간으로 업데이트
            prev_seg['end'] = current_seg['end']
        else:
            # 병합하지 않고 새 세그먼트로 추가
            merged_segments.append(current_seg)

# -----------------
# 3. 최종 출력 (중복 태그 수정 버전)
# -----------------

if merged_segments:
    # 첫 번째 세그먼트의 화자로 초기화
    current_speaker = merged_segments[0]['speaker']
    # 첫 번째 화자 정보와 타임스탬프를 먼저 출력
    print(f"[{merged_segments[0]['timestamp']}] [{current_speaker}]:", end="")

    for i in range(len(merged_segments)):
        segment = merged_segments[i]
        speaker = segment['speaker']
        text = segment['text']
        timestamp = segment['timestamp']

        if speaker == current_speaker:
            # 같은 화자이면 텍스트만 이어붙인다.
            print(f" {text}", end="")
        else:
            # 화자가 바뀌면
            # 1. 이전 화자의 발언을 마무리 (줄바꿈)
            print("\n")
            # 2. 새로운 화자 정보와 타임스탬프를 출력
            print(f"[{timestamp}] [{speaker}]: {text}", end="")
            # 3. 현재 화자 업데이트
            current_speaker = speaker
    
    # 모든 반복문이 끝난 후 마지막 줄바꿈
    print("\n")
else:
    print("처리할 발언이 없습니다.")
    
# 예시: JSON으로 결과 보기
# import json
# print(json.dumps(result["segments"], indent=2, ensure_ascii=False))

# 호출 : "http://127.0.0.1:5001/speaker?path=D:\test.mp4&key=11111&model=large-v3"
# 리턴 : "http://127.0.0.1/speaker_sucess.php?key=11111&path=D:\test_speaker.txt"