# Comments in English only to avoid font issues

import os
import json
import time
import sqlite3
import requests
import numpy as np
import sounddevice as sd
import subprocess
from vosk import Model, KaldiRecognizer

# ---------------- CONFIG ----------------

WAKE_WORD = "ipms oi"

DB_PATH = "/home/web/uiweb/database/ipms_paramater.db"

VOSK_MODEL_PATH = "/home/device/models/vosk-model-small-vn-0.4"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- ASR / TTS ----------------

vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)

tts = pyttsx3.init()
tts.setProperty("rate", 160)

def speak(text):
    print("AI:", text)
    subprocess.call(["espeak-ng", "-v", "vi", "-s", "150", text])

def listen_text():
    import numpy as np

    # many USB cards support 44100 by default
    HW_RATE = 44100

    recognizer = KaldiRecognizer(vosk_model, 16000)

    with sd.RawInputStream(
        samplerate=HW_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1
    ) as stream:
        while True:
            data, overflowed = stream.read(4000)

            # convert bytes to numpy int16
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # resample from HW_RATE -> 16000
            # simple ratio method (good enough for VOSK)
            ratio = 16000 / HW_RATE
            resampled = np.interp(
                np.arange(0, len(audio) * ratio, ratio),
                np.arange(0, len(audio)),
                audio
            ).astype(np.int16)

            if recognizer.AcceptWaveform(resampled.tobytes()):
                result = json.loads(recognizer.Result())
                txt = result.get("text", "")
                if txt:
                    print("USER:", txt)
                    return txt.lower()


# ---------------- SAVE USER STYLE ----------------

def save_user_phrase(text):
    with open("user_phrases.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

def load_user_phrases():
    if not os.path.exists("user_phrases.txt"):
        return ""
    with open("user_phrases.txt", "r", encoding="utf-8") as f:
        return f.read()

# ---------------- DB HELPERS ----------------

def sql_latest(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query)
    row = cur.fetchone()
    conn.close()
    return row

# temperature
def get_temp_now():
    try:
        row = sql_latest("""
            SELECT sensoripmst0, sensoripmst1
            FROM sensorlog20251225
            ORDER BY id DESC LIMIT 1
        """)
        return row[0]/10.0, row[1]/10.0
    except:
        return None, None

def get_temp_trend():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT sensoripmst0 FROM sensorlog20251225
            ORDER BY id DESC LIMIT 200
        """)
        rows = cur.fetchall()
        conn.close()

        vals = [r[0]/10.0 for r in rows if r[0] < 1000]
        if len(vals) < 20:
            return None

        x = np.arange(len(vals))
        slope = np.polyfit(x, vals, 1)[0]
        return slope
    except:
        return None

# ---------------- AIRCON ANALYSIS ----------------

def get_aircon_status():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT statusipms0, statusipms1,
                   relayipms0, relayipms1
            FROM relaylog20251225
            ORDER BY id DESC LIMIT 1
        """)
        s0, s1, c0, c1 = cur.fetchone()
        conn.close()

        return {
            "status": [s0, s1],
            "current": [c0/10.0, c1/10.0]
        }
    except:
        return None

def analyze_aircon_rotation(history=200):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT statusipms0, statusipms1
        FROM relaylog20251225
        ORDER BY id DESC LIMIT 200
    """)

    rows = cur.fetchall()
    conn.close()

    s0 = sum(1 for r in rows if r[0] == 2)
    s1 = sum(1 for r in rows if r[1] == 2)

    if abs(s0 - s1) < 10:
        return "Luân phiên máy lạnh hoạt động cân bằng, không lệch đáng kể."

    if s0 > s1:
        return "Máy lạnh số 0 chạy nhiều hơn số 1 đáng kể → nghi lỗi luân phiên hoặc máy 1 yếu."
    else:
        return "Máy lạnh số 1 chạy nhiều hơn số 0 đáng kể → nghi lỗi luân phiên hoặc máy 0 yếu."

def analyze_aircon_performance(status, current, room, out):
    result = []

    for i, (st, cur) in enumerate(zip(status, current)):

        if st == 2 and cur < 0.3:
            result.append(f"Máy lạnh {i} đang ON nhưng dòng thấp → block yếu hoặc thiếu gas.")
        if st == 1 and cur > 0.5:
            result.append(f"Máy lạnh {i} OFF nhưng vẫn có dòng → nghi rò điện/contactor dính.")

    if room and out and room - out > 10:
        result.append("Chênh nhiệt trong–ngoài lớn → tải lạnh cao hoặc hiệu suất suy giảm.")

    if not result:
        result.append("Hiệu suất máy lạnh tạm ổn, chưa ghi nhận dấu hiệu rõ ràng.")

    return result

# ---------------- DEEPSEA GENSET ANALYSIS ----------------

def read_latest_genset():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT reg0,reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8,reg9
            FROM genlog20251225
            ORDER BY id DESC LIMIT 1
        """)
        regs = cur.fetchone()
        conn.close()
        return regs
    except:
        return None

def parse_deepsea(regs):

    return {
        "voltage": regs[0]/10.0,
        "frequency": regs[1]/10.0,
        "current": regs[2]/10.0,
        "coolant": regs[4]/10.0,
        "oil": regs[5]/10.0,
        "fuel": regs[6]/10.0,
        "battery": regs[7]/10.0,
        "runtime": regs[8]/10.0
    }

def analyze_genset(data):
    res = []

    if data["frequency"] < 48 or data["frequency"] > 52:
        res.append("Tần số phát lệch chuẩn, cần kiểm tra governor/AVR.")

    if data["coolant"] > 95:
        res.append("Nhiệt nước làm mát cao, nguy cơ quá nhiệt.")

    if data["oil"] < 1.5:
        res.append("Áp suất dầu thấp, nên dừng máy và kiểm tra ngay.")

    if data["fuel"] < 20:
        res.append("Mức nhiên liệu dưới 20%, nên tiếp thêm sớm.")

    if data["battery"] < 11.5:
        res.append("Ắc quy đề yếu, nguy cơ đề máy không lên khi mất điện.")

    if data["runtime"] > 250:
        res.append("Số giờ chạy cao → nên lên lịch thay nhớt và lọc.")

    if not res:
        res.append("Máy phát đang hoạt động bình thường, không ghi nhận cảnh báo.")

    return res

# ---------------- AI CLOUD ENGINE ----------------

def ask_deepseek(system, user):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }

    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=payload,
        timeout=40
    )

    return r.json()["choices"][0]["message"]["content"]

# ---------------- SMART ANSWER ----------------

def build_all_facts():

    room, out = get_temp_now()
    slope = get_temp_trend()

    ac = get_aircon_status()

    if ac:
        ac_analysis = analyze_aircon_performance(
            ac["status"], ac["current"], room, out
        )
        rotation = analyze_aircon_rotation()
    else:
        ac_analysis = ["Không đọc được dữ liệu điều hòa."]
        rotation = "Không có dữ liệu luân phiên."

    genset_regs = read_latest_genset()
    if genset_regs:
        g_parsed = parse_deepsea(genset_regs)
        g_eval = analyze_genset(g_parsed)
    else:
        g_parsed = {}
        g_eval = ["Không đọc được dữ liệu máy phát."]

    return {
        "room_temp": room,
        "out_temp": out,
        "temp_trend": slope,
        "aircon": ac_analysis,
        "rotation": rotation,
        "genset": g_eval
    }

def smart_answer(question):

    save_user_phrase(question)

    facts = build_all_facts()

    user_style = load_user_phrases()

    system = f"""
Bạn là trợ lý AI kỹ thuật viễn thông cực kỳ thông minh.
Bạn nói chuyện giống kỹ thuật viên giàu kinh nghiệm, thân thiện, thực tế.

Bạn ưu tiên phong cách nói sau đây của người dùng thực tế:
{user_style}

Quy tắc:
- không bịa số liệu
- dùng đúng dữ liệu được cung cấp
- nói ngắn gọn, rõ, có chút Gen X, động viên tinh thần

Cấu trúc trả lời:
1) tổng quan tình trạng trạm
2) nhận định kỹ thuật thông minh
3) đề xuất hành động cụ thể
"""

    user = f"""
Câu hỏi của kỹ thuật viên: {question}

Dữ liệu hệ thống:
{json.dumps(facts, ensure_ascii=False)}
"""

    try:
        return ask_deepseek(system, user)
    except:
        return "Không truy cập được AI cloud. Nhưng tôi vẫn đang theo dõi dữ liệu tại chỗ."

# ---------------- MAIN LOOP ----------------

def main():

    speak("Hệ thống IPMS voice đã khởi động. Gọi tôi bằng câu: IPMS ơi.")

    while True:
        text = listen_text()

        if WAKE_WORD in text:
            speak("Tôi đây, bạn cần gì?")
            question = listen_text()

            reply = smart_answer(question)

            speak(reply)


if __name__ == "__main__":
    main()
