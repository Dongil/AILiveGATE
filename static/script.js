document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------------------------------
    // 1. 필요한 모든 HTML 요소들을 변수에 할당 (가장 먼저 수행)
    // -----------------------------------------------------------------
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileElem');
    const dropArea = document.getElementById('drop-area');
    const fileNameDisplay = document.getElementById('file-name-display');
    const instructionText = document.getElementById('drop-area-instruction'); 
    const submitBtn = document.getElementById('submitBtn');
    
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    const resultsContainer = document.getElementById('results-container');
    const statusMessage = document.getElementById('status-message');
    const resultView = document.getElementById('result-view');
    const resultContent = document.getElementById('result-content');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const downloadTxt = document.getElementById('download-txt');
    const downloadVtt = document.getElementById('download-vtt');
    
    let jobKey = null;
    let resultCache = { txt: null, vtt: null };

    // -----------------------------------------------------------------
    // 2. 핵심 기능 함수 정의
    // -----------------------------------------------------------------

    // 파일 이름 UI 업데이트 함수
    function updateFileNameDisplay(file) {
         if (file) {
            // 파일이 선택되면, 안내 문구는 숨기고 파일 이름만 보여줌
            instructionText.style.display = 'none';
            fileNameDisplay.textContent = file.name;
            fileNameDisplay.style.color = '#333';
            fileNameDisplay.style.fontWeight = '600';
        } else {
            // 파일 선택이 취소되면, 안내 문구를 다시 보여주고 기본 텍스트로 복귀
            instructionText.style.display = 'block';
            fileNameDisplay.textContent = '선택된 파일 없음';
            fileNameDisplay.style.color = 'var(--text-light-color)';
            fileNameDisplay.style.fontWeight = 'normal';
        }
    }

    // 브라우저 기본 동작 방지 함수
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // -----------------------------------------------------------------
    // 3. 이벤트 리스너 등록
    // -----------------------------------------------------------------

    // --- 파일 "클릭" 선택 시 ---
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        updateFileNameDisplay(file);
    });

    // --- 드래그 앤 드롭 관련 이벤트 ---
    // 전체 페이지에 대한 드래그 동작 방지 (새 탭에서 파일 열림 방지)
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults);
    });

    // 드롭 영역에 대한 시각적 피드백
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'));
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'));
    });

    // --- 파일을 "드롭"했을 때 (가장 중요) ---
    dropArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // 드롭된 파일을 숨겨진 input 요소에 할당
            fileInput.files = files; 
            // 파일 이름 UI 업데이트
            updateFileNameDisplay(files[0]);
        }
    });

    // -----------------------------------------------------------------
    // 4. 폼 제출 및 결과 처리 로직 (이하 코드는 이전과 동일)
    // -----------------------------------------------------------------
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files[0]) {
            alert('파일을 선택해주세요.');
            return;
        }
        
        resetUI();
        submitBtn.disabled = true;
        submitBtn.querySelector('.spinner').style.display = 'inline-block';
        submitBtn.querySelector('.btn-text').textContent = '업로드 중...';
        progressContainer.style.display = 'block';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model', document.getElementById('model').value);
        
        const optionalParams = ['threshold', 'min_duration_off', 'min_speakers', 'max_speakers'];
        optionalParams.forEach(id => {
            const element = document.getElementById(id);
            const placeholderValue = element.placeholder.split(': ')[1];
            
            // 값이 있으면 그 값을, 없으면 기본값을 폼에 추가
            formData.append(id, element.value || placeholderValue);
        });
        
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload-and-process', true);

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                progressBar.style.width = percentComplete + '%';
                progressText.textContent = `업로드: ${Math.round(percentComplete)}%`;
            }
        });

        xhr.onload = () => {
            submitBtn.querySelector('.btn-text').textContent = '처리 대기 중...';
            progressText.textContent = '업로드 완료, 서버 처리 대기 중입니다...';

            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                jobKey = response.key;
                resultsContainer.style.display = 'block';
                statusMessage.textContent = '✅ 작업이 대기열에 추가되었습니다. 처리가 시작되면 결과가 표시됩니다. (몇 분 정도 소요될 수 있습니다)';
                pollForResult();
            } else {
                handleError(`업로드 실패: ${xhr.statusText}`);
            }
        };

        xhr.onerror = () => {
            handleError('네트워크 오류가 발생했습니다.');
        };

        xhr.send(formData);
    });

    function pollForResult() {
        const interval = setInterval(async () => {
            try {
                // [수정] 파일 경로 대신 새로운 결과 확인 API를 호출합니다.
                const response = await fetch(`/job-result/${jobKey}`);
                
                if (response.ok) {
                    const result = await response.json();

                    if (result.status === 'completed') {
                        // 상태가 'completed'이면 폴링을 멈추고 결과를 표시합니다.
                        clearInterval(interval);
                        statusMessage.textContent = '🎉 처리가 완료되었습니다!';
                        displayResults(result.data); // 서버에서 받은 데이터(txt, vtt)를 전달합니다.
                    } else if (result.status === 'failed') {
                        // 상태가 'failed'이면 에러를 표시하고 멈춥니다.
                        clearInterval(interval);
                        handleError(`서버 처리 실패: ${result.data}`);
                    }
                    // 'processing' 상태이면 아무것도 하지 않고 다음 폴링을 기다립니다.
                    
                } else if (response.status === 404) {
                    // 작업 키를 찾을 수 없는 경우 (거의 발생하지 않음)
                    clearInterval(interval);
                    handleError('작업 ID를 찾을 수 없습니다. 페이지를 새로고침하세요.');
                } else {
                    // 그 외 서버 에러
                    clearInterval(interval);
                    handleError(`결과 확인 중 서버 오류 발생: ${response.statusText}`);
                }
            } catch (error) {
                clearInterval(interval);
                handleError('결과 확인 중 네트워크 오류가 발생했습니다.');
            }
        }, 5000); // 5초마다 확인
    }

    function displayResults(data) {
        // 이제 data는 { txt: "...", vtt: "..." } 형태의 객체입니다.
        // 더 이상 fetch로 파일을 가져올 필요가 없습니다.
        resultCache.txt = data.txt;
        resultCache.vtt = data.vtt;

        // --- 다운로드 링크를 Blob을 사용하여 동적으로 생성합니다 ---
        // TXT 다운로드
        try {
            const txtBlob = new Blob([resultCache.txt], { type: 'text/plain;charset=utf-8' });
            downloadTxt.href = URL.createObjectURL(txtBlob);
            downloadTxt.download = `${jobKey}_whisper.txt`;
        } catch (e) {
            console.error("TXT Blob 생성 실패:", e);
            downloadTxt.style.display = 'none';
        }

        // VTT 다운로드
        try {
            const vttBlob = new Blob([resultCache.vtt], { type: 'text/vtt;charset=utf-8' });
            downloadVtt.href = URL.createObjectURL(vttBlob);
            downloadVtt.download = `${jobKey}_whisper.vtt`;
        } catch (e) {
            console.error("VTT Blob 생성 실패:", e);
            downloadVtt.style.display = 'none';
        }
        // --- 여기까지 ---

        // 기본으로 txt 내용 표시
        resultContent.textContent = resultCache.txt;
        resultView.style.display = 'block';
        
        // 버튼 활성화
        submitBtn.disabled = false;
        submitBtn.querySelector('.spinner').style.display = 'none';
        submitBtn.querySelector('.btn-text').textContent = '회의록 생성 시작';
    }

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.dataset.target;
            resultContent.textContent = resultCache[target] || '내용을 불러올 수 없습니다.';
        });
    });
    
    function resetUI() {
        progressContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
        resultView.style.display = 'none';
        progressBar.style.width = '0%';
        progressText.textContent = '';
        statusMessage.textContent = '';
        resultContent.textContent = '';
        jobKey = null;
        resultCache = { txt: null, vtt: null };
    }
    
    function handleError(message) {
        alert(message);
        submitBtn.disabled = false;
        submitBtn.querySelector('.spinner').style.display = 'none';
        submitBtn.querySelector('.btn-text').textContent = '회의록 생성 시작';
    }
});