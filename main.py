# /main.py

import asyncio
import uuid
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

# 모듈화된 파일에서 필요한 것들 import
from config import (
    DEFAULT_MODEL_SIZE, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE,
    DEFAULT_DIARIZATION_THRESHOLD, DEFAULT_MIN_DURATION_OFF,
    DEFAULT_MIN_SPEAKERS, DEFAULT_MAX_SPEAKERS
)

from processor.tasks import process_video_and_callback, convert_video_to_audio, load_all_models
from app_state import job_results, job_queue # <<<--- 여기서 큐와 결과 딕셔너리를 import

worker_running = True       # 워커의 실행 상태를 제어하기 위한 플래그
worker_task = None          # 전역 변수로 선언

async def worker():
    """
    큐에서 작업을 하나씩 꺼내 순차적으로 처리하는 워커 함수
    """
    print("--- 작업 큐 워커 시작 ---")
    while worker_running:
        try:
            # 큐에서 작업 가져오기 (작업이 없으면 여기서 대기)
            task_details = await job_queue.get()
            
            # --- <<<--- 2. 작업 종류에 따른 분기 처리 ---
            task_name = task_details.get("task_name")
            task_params = task_details.get("params", {})

            # process_video_and_callback 함수는 동기 함수이므로,
            # asyncio 이벤트 루프를 막지 않도록 별도 스레드에서 실행
            
            if task_name == "diarize":
                # 기존 whisperx 작업
                await asyncio.to_thread(process_video_and_callback, **task_params)
            elif task_name == "convert":
                # 새로운 오디오 변환 작업
                await asyncio.to_thread(convert_video_to_audio, **task_params)
            else:
                print(f"알 수 없는 작업 타입입니다: {task_name}")

            # 작업이 끝났음을 큐에 알림
            job_queue.task_done()
        except Exception as e:
            print(f"워커에서 에러 발생: {e}")
            await asyncio.sleep(1) # 에러 발생 시 잠시 대기 후 계속

# --- <<<--- 2. 서버 시작/종료 시 워커 관리 ---
# --- <<<--- lifespan 이벤트 핸들러로 변경 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- 서버 시작 시 실행될 코드 --
    global worker_task
    print("서버 시작: AI 모델을 메모리에 로드합니다...")
    await asyncio.to_thread(load_all_models)
    worker_task = asyncio.create_task(worker())
    
    yield # 이 시점에서 애플리케이션이 실행됨

    # -- 서버 종료 시 실행될 코드 --
    global worker_running
    worker_running = False
    print("서버 종료: 워커를 안전하게 종료합니다...")
    await asyncio.sleep(2)
    if worker_task:
        worker_task.cancel()

# --- FastAPI 설정 ---
app = FastAPI(lifespan=lifespan) # FastAPI 앱 생성 시 lifespan을 등록

# Static 파일 (css, js)을 위한 디렉토리 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML 템플릿을 위한 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 업로드된 파일을 임시 저장할 디렉토리 생성
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- <<<--- 3. 새로운 라우터 추가 ---
@app.get("/audio_convert")
async def create_audio_convert_task(path: str, key: str, type: str):
    """영상 파일을 오디오로 변환하는 작업을 큐에 추가"""
    video_path = Path(path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found at: {path}")

    # 지원하는 타입인지 확인
    if type.lower() not in ['mp3', 'wav']:
        raise HTTPException(status_code=400, detail="Invalid type. 'type' must be 'mp3' or 'wav'.")

    task_details = {
        "task_name": "convert",
        "params": {
            "video_path": str(video_path),
            "key": key,
            "output_type": type.lower()
        }
    }
    await job_queue.put(task_details)

    return {
        "status": "queued",
        "message": f"오디오 변환 작업이 대기열에 추가되었습니다. (Key: {key})",
        "queue_size": job_queue.qsize()
    }

@app.get("/speaker")
async def create_speaker_task(
    path: str,
    key: str,
    model: str = DEFAULT_MODEL_SIZE, 
    # 선택적 쿼리 파라미터 추가
    threshold: float = Query(
        default=DEFAULT_DIARIZATION_THRESHOLD,
        description="Diarization clustering threshold (0.0 to 2.0). Lower is stricter."
    ),
    min_duration_off: float = Query(
        default=DEFAULT_MIN_DURATION_OFF,
        description="Minimum duration of non-speech segment in seconds."
    ),
    min_speakers: int = Query(
        default=DEFAULT_MIN_SPEAKERS,
        description="Approximate minimum number of participants attending the meeting. (2 to 25)"
    ),
    max_speakers: int = Query(
        default=DEFAULT_MAX_SPEAKERS,
        description="Approximate maximum number of participants attending the meeting. (2 to 25)"
    )
):
    """
    회의록 생성 작업을 큐에 추가하는 API 엔드포인트
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found at: {path}")
    
    # --- <<<--- 3. 작업을 큐에 넣기 ---
    # 백그라운드 태스크를 직접 실행하는 대신, 작업 정보를 딕셔너리로 만들어 큐에 넣는다.
    task_details = {
        "task_name": "diarize", # 작업 타입 명시
        "params" : {        
            "video_path": str(video_path),
            "key": key,
            "save_to_file": True,
            "model_name": model,
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
            # 받은 파라미터를 params 딕셔너리에 추가
            "diarization_params": {
                "threshold": threshold,
                "min_duration_off": min_duration_off,
                "min_speakers" : min_speakers,
                "max_speakers" : max_speakers
            }
        }
    }
    await job_queue.put(task_details)

    # 클라이언트(CMS)에는 즉시 응답
    return {
        "status": "queued",
        "message": f"화자분석 작업이 대기열에 추가되었습니다. (Key: {key})",
        "params_used": task_details["params"], # 어떤 파라미터가 사용되었는지 응답에 포함
        "queue_size": job_queue.qsize(), # 현재 대기 중인 작업 수
    }

# --- UI를 위한 새로운 엔드포인트들 ---
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    """메인 UI 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-and-process")
async def upload_and_process_video(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(...),
    min_duration_off: float = Form(...),
    min_speakers: int = Form(...),
    max_speakers: int = Form(...)
):
    """파일 업로드와 파라미터를 받아 작업을 큐에 추가합니다."""
    # 고유한 작업 키(key) 생성
    key = str(uuid.uuid4())
    
    # 업로드된 파일을 서버에 저장
    temp_path = UPLOAD_DIR / f"{key}_{file.filename}"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_details = {
        "task_name": "diarize",
        "params": {
            "video_path": str(temp_path),
            "key": key,
            "save_to_file":False,
            "model_name": model,
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
            "diarization_params": {
                "threshold": threshold,
                "min_duration_off": min_duration_off,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers
            }
        }
    }
    
    # 작업 상태를 'processing'으로 초기화
    job_results[key] = {"status": "processing", "data": None}
    await job_queue.put(task_details)

    return {
        "status": "queued",
        "message": "작업이 성공적으로 대기열에 추가되었습니다.",
        "key": key
    }
    
# --- <<<--- 3. UI용 새 라우터: 결과 확인 API 추가 ---
@app.get("/job-result/{key}")
async def get_job_result(key: str):
    """UI가 폴링하여 작업 상태와 결과를 가져가는 API"""
    result = job_results.get(key)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result
# --- 여기까지 ---

@app.get("/results/{key}/{file_type}")
async def get_result_file(key: str, file_type: str):
    """처리 완료된 결과 파일을 반환합니다. (폴링용)"""
    if file_type not in ["txt", "vtt"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    # tasks.py에서 정의한 출력 파일명 규칙과 일치해야 함
    # 예: "uploads/key_filename_whisper.txt"
    # tasks.py의 경로 생성 로직과 일관성을 위해 파일명 포맷을 맞춰야 합니다.
    # 여기서는 간단하게 key를 기반으로 파일을 찾습니다.
    
    # tasks.py의 출력 경로 생성 로직을 확인하고 일치시켜야 합니다.
    # 여기서는 tasks.py가 UPLOAD_DIR에 결과를 저장한다고 가정합니다.
    result_path = UPLOAD_DIR / f"{key}_whisper.{file_type}"
    
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found yet.")
    
    return FileResponse(result_path)

if __name__ == "__main__":
    # 서버 실행: python main.py
    # CMS나 다른 시스템에서 호출하려면 host를 "0.0.0.0"으로 설정해야 합니다.
    uvicorn.run(app, host="0.0.0.0", port=5001)