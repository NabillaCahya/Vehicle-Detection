import streamlit as st
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import tempfile
import math
import os
import matplotlib.pyplot as plt
import sqlite3
import json
import hashlib

MODEL_PATH = 'yolov8m.pt'
VEHICLE_CLASSES = ['bicycle', 'motorcycle', 'car', 'bus', 'truck']
VEHICLE_COLORS = {
    'car': ((70, 130, 180), (100, 180, 255)),
    'motorcycle': ((0, 191, 255), (100, 255, 255)),
    'bus': ((0, 102, 204), (0, 180, 255)),
    'truck': ((0, 153, 255), (100, 200, 255)),
    'bicycle': ((0, 102, 255), (100, 180, 255))
}
LIMITS = [100, 600, 1550, 600]
speed_factors = [1, 2, 4, 6, 8]
MAX_FILE_SIZE_MB = 1024

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            buf = f.read(65536)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()

conn = sqlite3.connect('analisis_kendaraan.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS progress (
        id INTEGER PRIMARY KEY,
        user_email TEXT,
        video_hash TEXT,
        video_path TEXT,
        frame_count INTEGER,
        counter_json TEXT,
        minute_counter_json TEXT,
        prev_y2_dict_json TEXT,
        totalcounts_json TEXT,
        status TEXT,
        last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
try:
    c.execute("ALTER TABLE progress ADD COLUMN user_email TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass

def load_last_progress(user_email):
    c.execute("SELECT video_hash, video_path, status FROM progress WHERE user_email=? ORDER BY last_update DESC LIMIT 1", (user_email,))
    row = c.fetchone()
    if row:
        return row[0], row[1], row[2]
    return None, None, None

def load_progress(user_email, video_hash, total_minutes):
    c.execute('SELECT frame_count, counter_json, minute_counter_json, prev_y2_dict_json, totalcounts_json, status FROM progress WHERE user_email=? AND video_hash=?', (user_email, video_hash))
    row = c.fetchone()
    if row:
        frame_count = row[0]
        counter = json.loads(row[1])
        minute_counter = json.loads(row[2])
        prev_y2_dict = json.loads(row[3])
        totalcounts = json.loads(row[4])
        status = row[5] if len(row) > 5 else "uploaded"
        if len(minute_counter) < total_minutes:
            for _ in range(total_minutes - len(minute_counter)):
                minute_counter.append({cls: 0 for cls in VEHICLE_CLASSES})
        elif len(minute_counter) > total_minutes:
            minute_counter = minute_counter[:total_minutes]
        return frame_count, counter, minute_counter, prev_y2_dict, totalcounts, status
    else:
        return 0, {cls: 0 for cls in VEHICLE_CLASSES}, [{cls: 0 for cls in VEHICLE_CLASSES} for _ in range(total_minutes)], {}, [], "uploaded"

def save_progress(user_email, video_hash, video_path, frame_count, counter, minute_counter, prev_y2_dict, totalcounts, status="analyzing"):
    c.execute('DELETE FROM progress WHERE user_email=? AND video_hash=?', (user_email, video_hash))
    c.execute('INSERT INTO progress (user_email, video_hash, video_path, frame_count, counter_json, minute_counter_json, prev_y2_dict_json, totalcounts_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            user_email,
            video_hash,
            video_path,
            frame_count,
            json.dumps(counter),
            json.dumps(minute_counter),
            json.dumps(prev_y2_dict),
            json.dumps(totalcounts),
            status
        )
    )
    conn.commit()

st.markdown("""
    <style>
    body, .main-container { font-family: 'Segoe UI', Arial, sans-serif; background: #e3f2fd; }
    .main-container {
        background: none !important;
        box-shadow: none !important;
        max-width: 1100px;
        margin: 32px auto 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .judul-box {
        background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%);
        border-radius: 32px;
        box-shadow: 0 8px 32px 0 #1976d244, 0 2px 16px 0 #64b5f644, 0 1.5px 8px 0 #fff4;
        padding: 32px 24px 32px 24px;
        margin-bottom: 36px;
        text-align: center;
        width: 100%;
        max-width: 800px;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 90px;
        flex-direction: column;
    }
    .judul {
        font-size: 2.5rem;
        font-weight: 900;
        color: #fff;
        margin: 0;
        width: 100%;
        text-align: center;
        letter-spacing: 1.2px;
        line-height: 1.18;
        text-shadow: 0 2px 8px #1976d255;
    }
    .subjudul {
        font-size: 1.1rem;
        font-weight: 500;
        color: #e3f2fd;
        margin-top: 8px;
        margin-bottom: 0;
        text-align: center;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 4px #1976d255;
    }
    .username-box {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px #1976d211;
        padding: 18px 32px 10px 32px;
        margin: 0 auto 24px auto;
        width: 100%;
        max-width: 400px;
        text-align: center;
    }
    .username-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 8px;
        letter-spacing: 0.2px;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        border-radius: 8px;
        border: 1.5px solid #1976d2;
        padding: 8px 12px;
    }
    .stFileUploader {
        background: #e3f2fd !important;
        border-radius: 16px !important;
        padding: 9px 18px 26px 18px !important;
        box-shadow: 0 2px 12px #1976d211 !important;
        margin-top: 0px !important;
        margin-bottom: 24px !important;
    }
    .stFileUploader label div[role="button"] {
        background: #1976d2 !important;
        color: #fff !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 24px;
        font-size: 16px;
        border: none;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .stFileUploader label div[role="button"]:hover {
        background: #1565c0 !important;
        color: #fff !important;
    }
    .stFileUploader .css-1b7of8t,
    .stFileUploader .st-dn {
        border: 2px solid #1976d2 !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin-top: 10px !important;
        background: #e3f2fd !important;
    }
    .stFileUploader .uploadedFile {
        border: 2px solid #1976d2 !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin-top: 10px !important;
        background: #e3f2fd !important;
    }
    .speed-box {
        background: #bbdefb;
        border-radius: 16px;
        box-shadow: 0 2px 12px #1976d211;
        padding: 22px 32px 18px 32px;
        margin: 0 auto 24px auto;
        width: 100%;
        max-width: 500px;
        text-align: center;
    }
    .speed-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 8px;
        letter-spacing: 0.2px;
    }
    .stRadio [role="radiogroup"] {
        justify-content: center;
        gap: 24px;
    }
    .stButton>button {
        background:#1976d2;
        color:#fff;
        font-weight:600;
        border-radius: 8px;
        padding: 10px 32px;
        font-size: 18px;
        margin-top: 16px;
        border: none;
        box-shadow: 0 2px 8px #1976d211;
        transition: background 0.2s;
    }
    .stButton>button:hover {background:#1565c0;}
    .hasil-analisis-box {
        background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%);
        border-radius: 18px;
        box-shadow: 0 4px 24px #1976d244;
        padding: 18px 0 18px 0;
        margin: 32px auto 32px auto;
        max-width: 600px;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hasil-analisis-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #fff;
        margin: 0;
        width: 100%;
        text-align: center;
        letter-spacing: 1px;
        line-height: 1.1;
        text-shadow: 0 2px 8px #1976d255;
    }
    .jumlah-kendaraan-box,
    .rekap-box,
    .grafik-box {
        background: linear-gradient(90deg, #e3f2fd 0%, #1976d2 100%);
        border-radius: 16px;
        box-shadow: 0 2px 12px #1976d244;
        padding: 22px 0 22px 0;
        margin: 36px auto 36px auto;
        max-width: 900px;
        text-align: center;
    }
    .jumlah-kendaraan-box h3,
    .rekap-box h3,
    .grafik-box h3 {
        color: #1976d2;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 18px;
        letter-spacing: 0.5px;
    }
    .label-jumlah {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 2px;
        color: #1976d2;
        font-family: 'Segoe UI', Arial, sans-serif;
        padding-left: 12px;
    }
    .isi-jumlah {
        font-size: 24px;
        font-weight: 700;
        color: #222;
        font-family: 'Segoe UI', Arial, sans-serif;
        padding-left: 12px;
    }
    .tabelku {
        border-radius: 12px;
        overflow-x: auto;
        margin: 0 auto;
        max-width: 98%;
    }
    .tabelku table {
        width: 100%;
        min-width: 600px;
        table-layout: fixed;
        border-collapse: collapse;
        margin: 0 auto;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .tabelku th, .tabelku td {
        width: 11% !important;
        text-align: center;
        padding: 12px 0;
        font-size: 18px;
        word-break: break-word;
    }
    .tabelku th:last-child,
    .tabelku td:last-child {
        width: 20% !important;
        min-width: 90px;
    }
    .tabelku th {
        background: #1976d2;
        color: #fff;
        font-weight: 600;
    }
    .tabelku td {
        background: #e3f2fd;
        font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }
    .tabelku tr:nth-child(even) td {
        background: #bbdefb;
    }
    .tabelku tr:hover td {
        background: #64b5f6;
        color: #fff;
    }
    .status-lancar { color: #388e3c !important; font-weight:700; }
    .status-macet { color: #d32f2f !important; font-weight:700; }
    .status-sedang { color: #fbc02d !important; font-weight:700; }
    .jumlah-kendaraan-scroll {
        overflow-x: auto;
        width: 100%;
    }
    .jumlah-kendaraan-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0 32px;
        min-width: 480px;
        max-width: 700px;
        margin-bottom: 8px;
    }
    @media (max-width: 600px) {
        .jumlah-kendaraan-grid {
            font-size: 13px !important;
            gap: 0 10px !important;
            min-width: 420px !important;
        }
    }
    @media (max-width: 600px) {
        .main-container {
            max-width: 100vw !important;
            padding: 0 2vw !important;
        }
        .judul-box {
            max-width: 98vw !important;
            padding: 18px 6vw 18px 6vw !important;
            min-height: 60px !important;
        }
        .judul {
            font-size: 1.3rem !important;
            line-height: 1.2 !important;
        }
        .subjudul {
            font-size: 0.9rem !important;
        }
        .hasil-analisis-box {
            max-width: 98vw !important;
            padding: 10px 2vw !important;
            min-height: 40px !important;
        }
        .hasil-analisis-title {
            font-size: 1.1rem !important;
        }
        .jumlah-kendaraan-box,
        .rekap-box,
        .grafik-box {
            max-width: 99vw !important;
            padding: 10px 2vw !important;
        }
        .tabelku table, .tabelku {
            font-size: 13px !important;
            max-width: 99vw !important;
        }
        .label-jumlah, .isi-jumlah {
            font-size: 13px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

col_logo, col_logo2 = st.columns([2, 10])
with col_logo:
    st.image("images/Logo ITS-Biru.png", width=110)
with col_logo2:
    st.image("images/1750992612695.png", width=70)

st.markdown(
    """
    <div style='
        background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%);
        border-radius:32px;
        box-shadow:0 8px 32px 0 #1976d244, 0 2px 16px 0 #64b5f644, 0 1.5px 8px 0 #fff4;
        margin-bottom:36px;
        margin-top:18px;
        padding:32px 24px 32px 24px;
        text-align:center;
        width:100%;
        max-width:800px;
        margin-left:auto;
        margin-right:auto;
    '>
        <div style='font-size:2.7rem; font-weight:900; color:#fff; letter-spacing:1.2px; text-shadow:0 2px 8px #1976d255; margin-bottom:10px;'>RapidPoint Flux</div>
        <div style='font-size:1.25rem; font-weight:600; color:#e3f2fd; text-shadow:0 1px 4px #1976d255;'>Monitoring sistem analisis arus dan memprediksi macet berbasis Ai</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="username-box">
        <div class="username-label">Masukkan Email Anda Untuk Memulai</div>
    </div>
    ''',
    unsafe_allow_html=True
)
user_email = st.text_input("Email", key="email_input")
if not user_email:
    st.warning("Silakan masukkan email terlebih dahulu untuk memulai.")
    st.stop()
user_email = user_email.strip().lower()

uploaded_file = st.file_uploader("", type=["mp4", "avi"])

if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'video_hash' not in st.session_state:
    st.session_state['video_hash'] = None
if 'status' not in st.session_state:
    st.session_state['status'] = None

if uploaded_file is not None:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"Ukuran file terlalu besar ({file_size_mb:.1f} MB). Maksimal {MAX_FILE_SIZE_MB} MB.")
        st.stop()
    st.success(f"File berhasil diupload ({file_size_mb:.1f} MB).")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    video_hash = get_file_hash(tfile.name)
    st.session_state['video_path'] = video_path
    st.session_state['video_hash'] = video_hash
    st.session_state['status'] = "uploaded"
    st.video(video_path)
    save_progress(user_email, video_hash, video_path, 0, {cls: 0 for cls in VEHICLE_CLASSES}, [], {}, [], status="uploaded")
else:
    if not st.session_state['video_path'] or not st.session_state['video_hash']:
        last_hash, last_path, last_status = load_last_progress(user_email)
        if last_hash and last_path and os.path.exists(last_path):
            st.session_state['video_path'] = last_path
            st.session_state['video_hash'] = last_hash
            st.session_state['status'] = last_status
        else:
            st.warning("Silakan upload video terlebih dahulu untuk memulai analisis.")
            st.stop()
    st.video(st.session_state['video_path'])

st.markdown(
    '''
    <div class="speed-box">
        <div class="speed-label">Pilih Kecepatan Analisis Video</div>
        <div style="color:#1976d2; font-size:0.97rem; margin-bottom:12px;">
            Semakin besar nilainya, proses analisis video akan semakin cepat.<br>
            Silakan pilih kecepatan terlebih dahulu, lalu klik <b>Mulai Analisis</b>.
        </div>
    ''',
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2])
with col1:
    speed = st.radio(
        "", speed_factors, index=0, horizontal=True, key="speed_radio"
    )
with col2:
    run_analysis = st.button("Mulai Analisis", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

frame_count, counter, minute_counter, prev_y2_dict, totalcounts, status = load_progress(user_email, st.session_state['video_hash'], 1)
auto_run_analysis = status in ("analyzing", "done")
if status == "uploaded" and not run_analysis:
    st.info("Video sudah diupload. Silakan klik 'Mulai Analisis' untuk memulai analisis.")

if run_analysis or auto_run_analysis:
    st.warning("Video sedang diproses, mohon tunggu hingga selesai...")

    with st.container():
        st.markdown(
            """
            <div class="hasil-analisis-box">
                <div class="hasil-analisis-title">Hasil Analisis Video</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(st.session_state['video_path'])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        duration_minutes = duration_seconds / 60
        total_minutes = int(duration_seconds // 60)
        if duration_seconds % 60 > 0:
            total_minutes += 1

        frame_count, counter, minute_counter, prev_y2_dict, totalcounts, _ = load_progress(user_email, st.session_state['video_hash'], total_minutes)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        stframe = st.empty()
        st.markdown("### Jumlah Kendaraan Terdeteksi")
        count_placeholder = st.empty()
        st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
        st.markdown("### Rekapitulasi Setiap Menit")
        stframe_table = st.empty()
        stframe_bar = st.empty()
        progress_bar = st.progress(0)

        last_bar_update_minute = -1

        while True:
            success, img = cap.read()
            if not success:
                break

            for _ in range(speed - 1):
                cap.read()
                frame_count += 1
            frame_count += 1

            img = cv2.resize(img, (frame_width, frame_height))
            result = model.track(
                img,
                stream=True,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.3,
                iou=0.5
            )

            elapsed_seconds = frame_count / fps
            elapsed_minutes = elapsed_seconds / 60
            current_minute = int(elapsed_minutes)
            if current_minute >= total_minutes:
                current_minute = total_minutes - 1

            for r in result:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    bbox = x1, y1, w, h
                    cls = int(box.cls[0])
                    if box.id is None:
                        continue  # skip jika tidak ada id
                    id = int(box.id[0])
                    if 0 <= cls < len(model.names):
                        currentClass = model.names[cls]
                    else:
                        continue
                    if currentClass in VEHICLE_COLORS:
                        color, _ = VEHICLE_COLORS[currentClass]
                        cvzone.cornerRect(img, bbox, l=9, rt=5, colorC=color)
                        cv2.line(img, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (25, 118, 210), 5)

                        garis_y = LIMITS[1]
                        prev_y2 = prev_y2_dict.get(str(id), y2)
                        if prev_y2 < garis_y and y2 >= garis_y:
                            if id not in totalcounts:
                                totalcounts.append(id)
                                counter[currentClass] += 1
                                minute_counter[current_minute][currentClass] += 1
                                cv2.line(img, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (255, 255, 255), 5)
                                cv2.circle(img, (x1 + w // 2, y1 + h // 2), 10, (25, 118, 210), cv2.FILLED)
                        prev_y2_dict[str(id)] = y2

            save_progress(user_email, st.session_state['video_hash'], st.session_state['video_path'], frame_count, counter, minute_counter, prev_y2_dict, totalcounts, status="analyzing")

            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img_rgb, channels="RGB")

            with count_placeholder:
                st.markdown(
                    """
                    <div class="jumlah-kendaraan-scroll">
                        <div class="jumlah-kendaraan-grid">
                            <div class="label-jumlah">Durasi</div>
                            <div class="label-jumlah">Sepeda</div>
                            <div class="label-jumlah">Motor</div>
                            <div class="label-jumlah">Mobil</div>
                            <div class="label-jumlah">Bus</div>
                            <div class="label-jumlah">Truk</div>
                        </div>
                        <div class="jumlah-kendaraan-grid">
                            <div class="isi-jumlah">{}</div>
                            <div class="isi-jumlah">{}</div>
                            <div class="isi-jumlah">{}</div>
                            <div class="isi-jumlah">{}</div>
                            <div class="isi-jumlah">{}</div>
                            <div class="isi-jumlah">{}</div>
                        </div>
                    </div>
                    """.format(
                        f"{minutes:02d}:{seconds:02d}",
                        counter['bicycle'],
                        counter['motorcycle'],
                        counter['car'],
                        counter['bus'],
                        counter['truck']
                    ),
                    unsafe_allow_html=True
                )

            if current_minute != last_bar_update_minute:
                last_bar_update_minute = current_minute

                interval = 5
                current_interval = (current_minute + 1) // interval
                num_intervals = (total_minutes + interval - 1) // interval

                html_table = '<div class="tabelku"><table>'
                html_table += (
                    "<tr>"
                    "<th>Menit</th><th>Sepeda</th><th>Motor</th><th>Mobil</th><th>Bus</th><th>Truk</th>"
                    "<th>Jumlah</th><th>Keterangan</th></tr>"
                )
                bar_labels = []
                bar_data = {k: [] for k in VEHICLE_CLASSES}
                bar_jumlah = []

                for idx in range(num_intervals):
                    start_min = idx * interval
                    end_min = min((idx + 1) * interval, total_minutes)
                    menit_label = f"{start_min+1}-{end_min}"
                    if current_minute + 1 >= (idx + 1) * interval:
                        row_sum = {cls: 0 for cls in VEHICLE_CLASSES}
                        for m in range(start_min, end_min):
                            for cls in VEHICLE_CLASSES:
                                row_sum[cls] += minute_counter[m][cls]
                        jumlah = sum(row_sum.values())
                        if jumlah > 600 * (end_min - start_min):
                            status_str = "Macet"
                            status_class = "status-macet"
                        elif jumlah < 400 * (end_min - start_min):
                            status_str = "Lancar"
                            status_class = "status-lancar"
                        else:
                            status_str = "Sedang"
                            status_class = "status-sedang"
                        html_table += (
                            f"<tr>"
                            f"<td>{menit_label}</td>"
                            f"<td>{row_sum['bicycle']}</td>"
                            f"<td>{row_sum['motorcycle']}</td>"
                            f"<td>{row_sum['car']}</td>"
                            f"<td>{row_sum['bus']}</td>"
                            f"<td>{row_sum['truck']}</td>"
                            f"<td>{jumlah}</td>"
                            f"<td class='{status_class}'>{status_str}</td>"
                            f"</tr>"
                        )
                        bar_labels.append(menit_label)
                        for k in VEHICLE_CLASSES:
                            bar_data[k].append(row_sum[k])
                        bar_jumlah.append(jumlah)
                    else:
                        html_table += (
                            f"<tr><td>{menit_label}</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
                        )
                html_table += "</table></div>"
                stframe_table.markdown(html_table, unsafe_allow_html=True)
                st.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)

                kendaraan_labels = ['Sepeda', 'Motor', 'Mobil', 'Bus', 'Truk']
                kendaraan_keys = ['bicycle', 'motorcycle', 'car', 'bus', 'truck']
                kendaraan_colors = ['#1976d2', '#64b5f6', '#90caf9', '#1565c0', '#42a5f5']
                bar_x = np.arange(len(bar_labels))
                fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                width = 0.15
                for idx_k, (k, color, label) in enumerate(zip(kendaraan_keys, kendaraan_colors, kendaraan_labels)):
                    data = bar_data[k]
                    x_pos = bar_x + (idx_k - 2) * width
                    ax_bar.bar(x_pos, data, width=width, label=label, color=color)
                ax_bar.set_xlabel("Menit ke-", fontsize=13)
                ax_bar.set_ylabel("Jumlah", fontsize=13)
                ax_bar.set_title("\n\nJumlah Kendaraan per Jenis per 5 Menit", fontsize=15, color="#1976d2", pad=24)
                ax_bar.set_xticks(bar_x)
                ax_bar.set_xticklabels(bar_labels)
                y_max = 10
                if len(bar_jumlah) > 0:
                    y_max = ((max(bar_jumlah) // 10) + 1) * 10
                ax_bar.set_ylim(0, y_max)
                ax_bar.set_yticks(range(0, y_max + 1, 10))
                ax_bar.legend(loc='upper right', fontsize=12)
                ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
                stframe_bar.pyplot(fig_bar)
                plt.close(fig_bar)

            if (frame_count >= total_frames):
                x_pos = [i+1 for i in range(total_minutes)]
                jumlah_list = [sum(minute_counter[i].values()) for i in range(total_minutes)]
                warna_list = []
                for jumlah in jumlah_list:
                    if jumlah > 80:
                        warna_list.append("#d32f2f")
                    elif jumlah < 60:
                        warna_list.append("#388e3c")
                    else:
                        warna_list.append("#fbc02d")
                fig_line, ax_line = plt.subplots(figsize=(10, 3))
                for i in range(len(jumlah_list)-1):
                    ax_line.plot(x_pos[i:i+2], jumlah_list[i:i+2], color=warna_list[i], linewidth=2, marker='o')
                ax_line.scatter(x_pos, jumlah_list, color=warna_list, s=80, zorder=5)
                ax_line.set_xlabel("Menit ke-")
                ax_line.set_ylabel("Jumlah Kendaraan")
                ax_line.set_title("Grafik Pertumbuhan Jumlah Kendaraan")
                ax_line.set_xticks(x_pos)
                y_max = ((max(jumlah_list) // 10) + 1) * 10 if max(jumlah_list) > 0 else 10
                ax_line.set_ylim(0, y_max)
                ax_line.set_yticks(range(0, y_max + 1, 10))
                for x, y, warna in zip(x_pos, jumlah_list, warna_list):
                    ax_line.text(x, y, str(y), ha='center', va='bottom', fontsize=10, color=warna)
                st.pyplot(fig_line)

            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)

        cap.release()

        save_progress(user_email, st.session_state['video_hash'], st.session_state['video_path'], frame_count, counter, minute_counter, prev_y2_dict, totalcounts, status="done")
        st.success("Analisis selesai!")

        st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)