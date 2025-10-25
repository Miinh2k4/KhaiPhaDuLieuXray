@echo off
REM ================================
REM 1) Tạo môi trường ảo (nếu chưa có)
REM ================================
IF NOT EXIST .venv (
   echo [INFO] Dang tao virtual environment...
   python -m venv .venv
)

REM ================================
REM 2) Kich hoat moi truong ao
REM ================================
echo [INFO] Kich hoat virtual environment...
call .venv\Scripts\activate

REM ================================
REM 3) Cai thu vien neu chua co
REM    (Neu da cai roi thi pip bo qua)
REM ================================
echo [INFO] Cai dat cac thu vien can thiet...
pip install --upgrade pip
pip install streamlit tensorflow opencv-python pillow numpy matplotlib

REM ================================
REM 4) Chay ung dung Streamlit
REM    (mac dinh file ten 'appfixedv7ok.py')
REM ================================
echo ----------------------------------------
echo [INFO] Dang chay ung dung Streamlit...
echo ----------------------------------------
streamlit run appfixedv7ok.py

REM ================================
REM 5) Giu cua so mo de xem log
REM ================================
echo ----------------------------------------
echo [DONE] Bam phim bat ky de thoat.
pause
