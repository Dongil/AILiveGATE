# /processor/tasks.py

import whisperx
import gc
import ffmpeg
import requests
from pathlib import Path
from whisperx.diarize import DiarizationPipeline
import traceback

# 프로젝트 루트의 config.py에서 설정값 가져오기
from config import SPEAKER_CALLBACK_URL, MERGE_THRESHOLD_SECONDS, SHORT_SEGMENT_WORD_COUNT, HF_TOKEN, AUDIO_CALLBACK_URL
from app_state import job_results

# --- <<<--- 1. 모델을 담을 전역 변수 선언 ---
MODELS = {
    "asr": None,
    "align": None,
    "diarize": None
}

def load_all_models(model_name="large-v3", device="cuda", compute_type="float16"):
    """
    서버 시작 시 모든 AI 모델을 한 번만 로드하여 전역 변수에 저장합니다.
    """
    print("--- 공유 AI 모델 로딩 시작 ---")
 
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
    save_to_file: bool,
    model_name: str,
    device: str,
    compute_type: str,
    # --- <<<--- 튜닝 파라미터를 받을 딕셔너리 추가 ---
    diarization_params: dict     # diarization_params는 {'threshold': 0.55, 'min_duration_off': 0.8} 형태
):
    """
    영상 파일을 받아 회의록과 VTT 파일을 생성하는 태스크 (튜닝 파라미터 적용)
    """
    print(f"--- 작업 시작 (Key: {key}) ---")
    print(f"영상 파일: {video_path}")
    print(f"모델: {model_name}, 장치: {device}, 타입: {compute_type}")
    print(f"화자 분리 파라미터: {diarization_params}")

    # key 값을 파일명으로 사용
    output_path = Path(video_path).parent / key
    
    output_txt_path = output_path.with_name(f"{output_path.name}_whisper.txt")
    output_vtt_path = output_path.with_name(f"{output_path.name}_whisper.vtt")

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
        print(f"  - 파라미터 적용: {diarization_params}")
         
        # 2-3-1. 파이프라인 내부 속성 값을 직접 변경합니다.
        #    'pipeline'이 아니라 'model' 속성을 통해 pyannote 객체에 접근합니다.
        if 'threshold' in diarization_params:
            diarize_model.model.clustering.threshold = diarization_params['threshold']
        if 'min_duration_off' in diarization_params:
            diarize_model.model.segmentation.min_duration_off = diarization_params['min_duration_off']

        # 2-3-2. 파라미터가 수정된 모델로 화자 분리를 실행합니다.
        diarize_segments = diarize_model(
            audio, 
            min_speakers=diarization_params['min_speakers'],
            max_speakers=diarization_params['max_speakers']
        )
        
        # 2-3-3. assign_word_speakers 호출
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("화자 분리 완료.")
                
        # --- 3. 후처리 및 파일 저장 ---
        print("3. 후처리 및 파일 저장 중...")
        
        final_transcript = generate_formatted_transcript(result)
        vtt_content = generate_vtt_content(result)

        if save_to_file:
            # API 호출의 경우: 파일로 저장하고 콜백 전송
            output_path = Path(video_path)
            output_txt_path = output_path.parent / f"{output_path.stem}_whisper.txt"
            output_vtt_path = output_path.parent / f"{output_path.stem}_whisper.vtt"

            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_transcript)
            print(f"회의록 파일 저장 완료: {output_txt_path}")
            
            with open(output_vtt_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            print(f"VTT 파일 저장 완료: {output_vtt_path}")

            send_completion_callback(
                url=SPEAKER_CALLBACK_URL,
                success=True, 
                key=key, 
                path=str(output_txt_path)
            )
        else:
            # 웹 UI 호출의 경우: 인메모리 딕셔너리에 저장
            job_results[key] = {
                "status": "completed",
                "data": {
                    "txt": final_transcript,
                    "vtt": vtt_content
                }
            }
            print(f"작업 결과 메모리에 저장 완료 (Key: {key})")

    except Exception as e:
        # --- <<<--- 이 부분을 수정하여 상세한 에러 로그를 얻습니다. ---
        # 1. 전체 에러 트레이스백을 콘솔에 출력
        print("---!!! 작업 중 심각한 오류 발생 !!!---")
        traceback.print_exc()
        print("------------------------------------")
        
        # 2. 에러 메시지를 더 상세하게 생성
        error_message = f"작업 실패 (Key: {key}): {type(e).__name__} - {e}"
        print(error_message)

        # --- <<<--- 이 부분 로직 수정 ---
        if save_to_file:
            # 파일 저장 모드에서 에러 발생 시에만 파일에 에러 내용 기록
            # output_txt_path가 try 블록 초반에 정의되므로 사용 가능
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(error_message)
            
            send_completion_callback(
                url=SPEAKER_CALLBACK_URL,
                success=False,
                key=key,
                path=str(output_txt_path),
                error=str(e)
            )
        else:
            # UI 모드에서는 job_results에 에러 상태 기록
            job_results[key] = {"status": "failed", "data": error_message}
    finally:
        # 임시 오디오 파일 삭제
        if 'audio_path_obj' in locals() and audio_path_obj.exists():
            audio_path_obj.unlink()
        print(f"--- 작업 종료 (Key: {key}) ---")
        
def generate_formatted_transcript(result: dict) -> str:
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