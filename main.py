# /main.py

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from pathlib import Path
import uvicorn

# 모듈화된 파일에서 필요한 것들 import
from config import DEFAULT_MODEL_SIZE, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE
from processor.tasks import process_video_and_callback, load_all_models

# --- <<<--- 1. 작업 큐와 워커 설정 ---
job_queue = asyncio.Queue() # 비동기 작업 큐 생성
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
            
            # process_video_and_callback 함수는 동기 함수이므로,
            # asyncio 이벤트 루프를 막지 않도록 별도 스레드에서 실행
            await asyncio.to_thread(
                process_video_and_callback,
                **task_details
            )
            
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

app = FastAPI(lifespan=lifespan) # FastAPI 앱 생성 시 lifespan을 등록

@app.get("/speaker")
async def create_speaker_task(
    path: str,
    key: str,
    model: str = DEFAULT_MODEL_SIZE
):
    """
    회의록 생성 작업을 큐에 추가하는 API 엔드포인트
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found at: {path}")

    output_txt_path = video_path.parent / f"{video_path.stem}_speaker.txt"
    
    # --- <<<--- 3. 작업을 큐에 넣기 ---
    # 백그라운드 태스크를 직접 실행하는 대신, 작업 정보를 딕셔너리로 만들어 큐에 넣는다.
    task_details = {
        "video_path": str(video_path),
        "key": key,
        "model_name": model,
        "device": DEFAULT_DEVICE,
        "compute_type": DEFAULT_COMPUTE_TYPE
    }
    await job_queue.put(task_details)
    # --- 여기까지 ---

    # 클라이언트(CMS)에는 즉시 응답
    return {
        "status": "queued",
        "message": f"작업이 대기열에 추가되었습니다. (Key: {key})",
        "queue_size": job_queue.qsize(), # 현재 대기 중인 작업 수
        "output_path": str(output_txt_path)
    }

@app.get("/")
def read_root():
    return {"message": "AI 속기록 생성 서버가 실행 중입니다."}

if __name__ == "__main__":
    # 서버 실행: python main.py
    # CMS나 다른 시스템에서 호출하려면 host를 "0.0.0.0"으로 설정해야 합니다.
    uvicorn.run(app, host="0.0.0.0", port=5001)