import cv2
import requests
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import os

# ====== CONFIGURAÇÃO ======
USAR_VIDEO = True  # True = vídeo local | False = ESP32-CAM
VIDEO_PATH = "video.mp4"  # Nome do seu vídeo
capture_url = "http://192.168.1.65/capture"

model_path = "yolov8n.pt"
linha_fila = 400

model = YOLO(model_path)

# rastreamento e tempo de permanência
tempo_entrada = {}
tempo_espera = []
API_URL = "http://localhost:3000/api/fila"

if USAR_VIDEO:
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Vídeo não encontrado: {VIDEO_PATH}")
        print("💡 Coloque o vídeo na pasta backend/")
        exit()
    print(f"✅ Carregando vídeo: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Erro ao abrir o vídeo")
        exit()
else:
    print("✅ Sistema iniciado! Usando ESP32-CAM...")
    cap = None

while True:
    try:
        if USAR_VIDEO:
            ret, frame = cap.read()
            if not ret:
                print("🔄 Vídeo terminou, reiniciando...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            resp = requests.get(capture_url, timeout=5)
            frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            continue

        height, width, _ = frame.shape
        results = model(frame, verbose=False)

        pessoas = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # pessoa
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pessoas.append((x1, y1, x2, y2))
                    # aplica blur no rosto (região superior)
                    rosto = frame[y1:y1 + int((y2 - y1) / 3), x1:x2]
                    if rosto.size > 0:
                        rosto = cv2.GaussianBlur(rosto, (45, 45), 30)
                        frame[y1:y1 + int((y2 - y1) / 3), x1:x2] = rosto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        pessoas_na_fila = len(pessoas)
        agora = time.time()

        # simulação de tempo de espera: cada pessoa leva ~20s para sair
        tempo_esperado = pessoas_na_fila * 20

        data = {
            "pessoas": pessoas_na_fila,
            "tempo_medio_espera": tempo_esperado,
            "timestamp": datetime.now().isoformat()
        }

        requests.post(API_URL, json=data, timeout=3)
        print(f"➡️ Enviado: {data}")

        cv2.imshow("Fila Anonimizada", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not USAR_VIDEO:
            time.sleep(2)  # Delay apenas para ESP32

    except Exception as e:
        print("Erro:", e)
        time.sleep(1)

if USAR_VIDEO:
    cap.release()
cv2.destroyAllWindows()
