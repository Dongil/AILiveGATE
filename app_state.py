# /app_state.py
import asyncio

# UI용 결과 저장을 위한 전역 딕셔너리
job_results = {}

# 작업 큐
job_queue = asyncio.Queue()