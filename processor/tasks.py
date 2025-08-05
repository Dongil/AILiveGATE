# /processor/tasks.py

import whisperx
import gc
import ffmpeg
import requests
from pathlib import Path
from whisperx.diarize import DiarizationPipeline

# 프로젝트 루트의 config.py에서 설정값 가져오기
from config import SPEAKER_CALLBACK_URL, MERGE_THRESHOLD_SECONDS, SHORT_SEGMENT_WORD_COUNT, HF_TOKEN, AUDIO_CALLBACK_URL
                    
# --- <<<--- 1. 모델을 담을 전역 변수 선언 ---
MODELS = {
    "asr": None,
    "align": None,
    "diarize": None
}
# --- 여기까지 ---

def load_all_models(model_name="large-v3", device="cuda", compute_type="float16"):
    """
    서버 시작 시 모든 AI 모델을 한 번만 로드하여 전역 변수에 저장합니다.
    """
    print("--- 모든 AI 모델 로딩 시작 ---")
    
    # 1. ASR 모델 로드
    print(f"ASR 모델({model_name}) 로딩 중...")
    MODELS["asr"] = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    # 2. Align 모델 로드 (언어는 'ko'로 고정)
    print("Align 모델 로딩 중...")
    model_a, metadata = whisperx.load_align_model(language_code="ko", device=device)
    MODELS["align"] = {"model": model_a, "metadata": metadata}
    
    # 3. Diarization 모델 로드
    print("Diarization 모델 로딩 중...")
    # hf_token은 huggingface-cli login을 통해 자동으로 사용됩니다.
    MODELS["diarize"] = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    
    print("--- 모든 AI 모델 로딩 완료 ---")

# --- <<<--- 1. VTT 생성 함수 추가 ---
def format_vtt_time(seconds: float) -> str:
    """초(float)를 VTT 타임스탬프 형식 (HH:MM:SS.mmm)으로 변환"""
    # ... (이전과 동일한 시간 변환 로직, 밀리초 추가) ...
    hours, remainder = divmod(int(seconds), 3600)
    minutes, remainder = divmod(remainder, 60)
    secs = int(remainder)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def generate_vtt_content(result: dict) -> str:
    """whisperx 결과물을 받아 WebVTT 형식의 문자열을 생성"""
    if 'segments' not in result or not result['segments']:
        return "WEBVTT\n\n"

    lines = ["WEBVTT", ""] # VTT 파일 헤더

    for segment in result['segments']:
        start_time = format_vtt_time(segment['start'])
        end_time = format_vtt_time(segment['end'])
        
        # 화자 정보를 자막에 포함시킬 수 있음 (선택적)
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment.get('text', '').strip()

        # 자막 큐 추가
        lines.append(f"{start_time} --> {end_time}")
        lines.append(f"<{speaker}> {text}") # 예: <SPEAKER_01> 안녕하세요.
        lines.append("") # 각 큐 사이에 빈 줄 추가

    return "\n".join(lines)
# --- 여기까지 ---

# --- <<<--- 1. 새로운 오디오 변환 작업 함수 추가 ---
def convert_video_to_audio(
    video_path: str,
    key: str,
    output_type: str # 'mp3' or 'wav'
):
    """
    영상 파일을 지정된 오디오 포맷으로 변환하는 태스크.
    """
    print(f"--- 오디오 변환 작업 시작 (Key: {key}) ---")
    print(f"영상 파일: {video_path}, 변환 타입: {output_type}")
    
    # 출력 파일 경로 생성 (예: D:\test.mp3)
    output_audio_path = Path(video_path).with_suffix(f'.{output_type}')

    try:
        if output_type == 'mp3':
            # Clova Speech용 MP3 설정
            ffmpeg.input(video_path).output(
                str(output_audio_path),
                acodec='libmp3lame', # MP3 인코더
                audio_bitrate='192k',
                ac=1, # Mono 채널
                ar='16000' # 16kHz 샘플링 레이트
            ).run(overwrite_output=True, quiet=True)
        elif output_type == 'wav':
            # Google Speech-to-Text용 WAV 설정
            ffmpeg.input(video_path).output(
                str(output_audio_path),
                acodec='pcm_s16le', # 16-bit PCM 인코딩
                ac=1, # Mono 채널
                ar='16000' # 16kHz 샘플링 레이트
            ).run(overwrite_output=True, quiet=True)
        else:
            raise ValueError(f"지원하지 않는 오디오 타입입니다: {output_type}")
            
        print(f"오디오 파일 변환 완료: {output_audio_path}")
        
        # 완료 콜백 전송
        send_completion_callback(
            url=AUDIO_CALLBACK_URL,
            success=True,
            key=key,
            path=str(output_audio_path),
            extra_params={'type': output_type}
        )
            
    except Exception as e:
        error_message = f"오디오 변환 작업 실패 (Key: {key}): {e}"
        print(error_message)
        send_completion_callback(
            url=AUDIO_CALLBACK_URL,
            success=False,
            key=key,
            path=str(output_audio_path),
            error=str(e),
            extra_params={'type': output_type}
        )
    finally:
        print(f"--- 오디오 변환 작업 종료 (Key: {key}) ---")
# --- 여기까지 ---

def process_video_and_callback(
    video_path: str,
    key: str,
    model_name: str,
    device: str,
    compute_type: str
):
    """
    영상 파일을 받아 회의록과 VTT 파일을 생성하고, 콜백을 호출하는 메인 태스크.
    """
    print(f"--- 작업 시작 (Key: {key}) ---")
    print(f"영상 파일: {video_path}")
    print(f"모델: {model_name}, 장치: {device}, 타입: {compute_type}")

    # --- <<<--- 출력 파일 경로 설정 추가 ---
    output_path = Path(video_path)
    
    output_txt_path = output_path.parent / f"{output_path.stem}_whisper.txt"    
    output_vtt_path = output_path.parent / f"{output_path.stem}_whisper.vtt"

    try:
        # --- 1. 오디오 추출 ---
        audio_path_obj = Path(video_path).with_suffix('.wav')
        audio_path = str(audio_path_obj)
        
        print("1. 오디오 추출 중...")
        ffmpeg.input(video_path).output(
            audio_path, acodec='pcm_s16le', ac=1, ar='16000'
        ).run(overwrite_output=True, quiet=True)
        print("오디오 추출 완료.")

        # --- <<<--- 2. 전역 모델 재사용 ---
        print("2. 로드된 모델을 사용하여 처리 시작...")
        
        # 모델을 새로 로드하는 대신, 전역 변수에서 가져옵니다.
        asr_model = MODELS["asr"]
        align_model_data = MODELS["align"]
        diarize_model = MODELS["diarize"]

        if not all([asr_model, align_model_data, diarize_model]):
            # 모델이 로드되지 않은 경우 에러 처리
            raise RuntimeError("모델이 정상적으로 로드되지 않았습니다. 서버를 재시작하세요.")

        audio = whisperx.load_audio(audio_path)

        # 2-1. ASR
        print("   - 음성 인식(ASR) 진행 중...")
        result = asr_model.transcribe(audio, language="ko", batch_size=16)
        
        # 2-2. Align
        print("   - 타임스탬프 정렬 중...")
        result = whisperx.align(
            result["segments"], 
            align_model_data["model"], 
            align_model_data["metadata"], 
            audio, 
            device, 
            return_char_alignments=False
        )

        # 2-3. Diarize
        print("   - 화자 분리 진행 중...")
        # diarize_segments = diarize_model(audio) # 화자 수 힌트 제공 가능
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=25)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("화자 분리 완료.")
        
        # --- 3. 후처리 및 파일 저장 ---
        print("3. 후처리 및 파일 저장 중...")
        
        # 3-1. 회의록 텍스트(.txt) 파일 생성
        # (기존의 후처리 및 병합 로직은 그대로 사용)
        final_transcript = generate_formatted_transcript(result)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(final_transcript)
        print(f"회의록 파일 저장 완료: {output_txt_path}")
        
        # 3-2. WebVTT(.vtt) 파일 생성
        vtt_content = generate_vtt_content(result)
        with open(output_vtt_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        print(f"VTT 파일 저장 완료: {output_vtt_path}")
        # --- 여기까지 ---

        # --- 4. 콜백 URL 호출 ---
        send_completion_callback(
            url=SPEAKER_CALLBACK_URL,
            success=True, 
            key=key, 
            path=str(output_txt_path)
        )

    except Exception as e:
        error_message = f"작업 실패 (Key: {key}): {e}"
        print(error_message)
        # 실패 시에도 콜백을 보내서 CMS가 상태를 알 수 있게 함 (선택적)
        # 실패 내용을 파일에 쓸 수도 있음
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(error_message)
        send_completion_callback(
            url=SPEAKER_CALLBACK_URL,
            success=False, 
            key=key, 
            path=str(output_txt_path), 
            error=str(e)
        )
    finally:
        # 임시 오디오 파일 삭제
        if 'audio_path_obj' in locals() and audio_path_obj.exists():
            audio_path_obj.unlink()
        print(f"--- 작업 종료 (Key: {key}) ---")


def generate_formatted_transcript(result: dict) -> str:
    # <<<--- 이 부분이 누락되었습니다. ---
    # 1. 초기 세그먼트 정리 및 형식 변환
    processed_segments = []
    if "segments" in result and result["segments"]:
        for seg in result["segments"]:
            if 'speaker' not in seg:
                seg['speaker'] = 'UNKNOWN'
            
            start_time = int(seg['start'])
            
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

    # 텍스트를 담을 리스트
    output_lines = []

    if merged_segments:
        current_speaker = merged_segments[0]['speaker']
        output_lines.append(f"[{merged_segments[0]['timestamp']}] [{current_speaker}]:")

        for i in range(len(merged_segments)):
            segment = merged_segments[i]
            speaker = segment['speaker']
            text = segment['text']
            timestamp = segment['timestamp']
           
            if speaker == current_speaker:
                output_lines[-1] += f" {text}"
            else:
                output_lines.append(f"\n[{timestamp}] [{speaker}]: {text}")
                current_speaker = speaker
    else:
        output_lines.append("처리할 발언이 없습니다.")

    return "".join(output_lines)

def send_completion_callback(url: str, success: bool, key: str, path: str, error: str = "", extra_params: dict = None):
    """CMS에 작업 완료/실패를 알리는 범용 콜백 함수"""
    params = {'key': key, 'path': path}
    if not success:
        params['error'] = error
    if extra_params:
        params.update(extra_params) # 추가 파라미터 병합 (type=mp3 등)

    try:
        print(f"콜백 전송 시도: {url} with params {params}")
        response = requests.get(url, params=params, timeout=10)        
        response.raise_for_status()
        print(f"콜백 전송 성공 (Key: {key})")
    except requests.RequestException as e:
        print(f"콜백 전송 실패 (Key: {key}): {e}")