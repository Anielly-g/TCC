"""
Sistema de Contagem de Pessoas - COMPARAÇÃO 3 DETECTORES OTIMIZADA
✅ HOG + SVM, YOLOv8n e MediaPipe Pose
✅ Skip adaptativo baseado na velocidade
✅ Anonimização apenas em frames processados
✅ Limite configurável (padrão: 10 minutos)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import json
from datetime import datetime
from tkinter import Tk, filedialog

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False
    print("⚠️  AVISO: MediaPipe não disponível")

class ThreeDetectorComparison:
    def __init__(self, line_position=0.5, threshold=30, memory_frames=10):
        """
        Comparação otimizada de 3 detectores
        """
        self.line_position = line_position
        self.threshold = threshold
        self.memory_frames = memory_frames
        
        print("\n🔧 CARREGANDO DETECTORES...")
        
        # DETECTOR 1: HOG + SVM
        print("   [1/3] HOG + SVM...")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # DETECTOR 2: YOLOv8n
        print("   [2/3] YOLOv8n...")
        self.yolo = YOLO('yolov8n.pt')
        
        # DETECTOR 3: MediaPipe Pose
        print("   [3/3] MediaPipe Pose...")
        if _MEDIAPIPE_AVAILABLE:
            self.mp = mp
            self.mp_pose = self.mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = self.mp.solutions.drawing_utils
        else:
            self.mp_pose = None
            print("   ❌ MediaPipe não disponível")
        
        print("   ✅ Todos os detectores carregados!\n")
        
        # Contadores separados
        self.count_hog = 0
        self.count_yolo = 0
        self.count_mediapipe = 0
        
        # Histórico de centroides
        self.prev_centroids_hog = []
        self.prev_centroids_yolo = []
        self.prev_centroids_mediapipe = []
        
        # Anti-recontagem
        self.recent_crossings_hog = deque(maxlen=memory_frames * 10)
        self.recent_crossings_yolo = deque(maxlen=memory_frames * 10)
        self.recent_crossings_mediapipe = deque(maxlen=memory_frames * 10)
        
        self.max_move = 600
        self.frame_count = 0
        self.detections_log = []
        
    def analyze_scene_velocity(self, cap, num_samples=90):
        """
        Analisa velocidade do cenário para determinar skip ideal
        Garante detecções suficientes antes do cruzamento
        """
        print("\n🔍 ANALISANDO VELOCIDADE DO CENÁRIO...")
        print("   Processando primeiros 3 segundos do vídeo...")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        velocities = []
        prev_detections = []
        
        for i in range(min(num_samples, fps * 3)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Usar YOLO para análise (mais rápido)
            detections = self.detect_yolo(frame)
            current_centroids = [d['centroid'] for d in detections]
            
            # Calcular velocidades
            if prev_detections:
                for curr_cx, curr_cy in current_centroids:
                    for prev_cx, prev_cy in prev_detections:
                        dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                        if dist < 100:  # Mesma pessoa
                            velocities.append(dist)
                            break
            
            prev_detections = current_centroids
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Voltar ao início
        
        if not velocities:
            print("   ⚠️  Nenhuma pessoa detectada nos primeiros 3s")
            print("   📌 Usando skip conservador = 60 frames")
            return 60, "padrão (sem movimento detectado)"
        
        # Estatísticas
        avg_velocity = np.mean(velocities)
        max_velocity = np.percentile(velocities, 95)  # 95º percentil
        
        print(f"\n   📊 ANÁLISE COMPLETA:")
        print(f"   📏 Velocidade média: {avg_velocity:.1f} pixels/frame")
        print(f"   🏃 Velocidade máxima (95%): {max_velocity:.1f} pixels/frame")
        print(f"   📈 Amostras coletadas: {len(velocities)}")
        
        # Determinar skip CONSERVADOR para garantir 5+ detecções
        # Regra: pessoa deve ser detectada em pelo menos 5 frames antes de cruzar
        if max_velocity < 8:
            skip = 90  # Pessoas quase paradas
            tipo = "pessoas paradas/muito lentas"
        elif max_velocity < 15:
            skip = 60  # Caminhada lenta
            tipo = "caminhada lenta/normal"
        elif max_velocity < 25:
            skip = 40  # Caminhada normal
            tipo = "caminhada normal"
        elif max_velocity < 40:
            skip = 25  # Caminhada rápida
            tipo = "caminhada rápida"
        else:
            skip = 15  # Correndo
            tipo = "corrida/movimento rápido"
        
        print(f"\n   🎯 SKIP RECOMENDADO: {skip} frames")
        print(f"   📝 Cenário detectado: {tipo}")
        print(f"   ⚡ Economia de processamento: {(skip-1)/skip*100:.0f}%")
        print(f"   ✅ Garante 5+ detecções por pessoa antes do cruzamento\n")
        
        return skip, tipo
    
    def anonymize_faces(self, frame):
        """Blur otimizado em faces"""
        if FACE_CASCADE.empty():
            return frame
        
        h, w = frame.shape[:2]
        scale = 0.5  # Processa em metade da resolução
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(15, 15)
        )
        
        for (x, y, w_face, h_face) in faces:
            # Reescalar para resolução original
            x = int(x / scale)
            y = int(y / scale)
            w_face = int(w_face / scale)
            h_face = int(h_face / scale)
            
            x_end = min(x + w_face, w)
            y_end = min(y + h_face, h)
            x_start = max(0, x)
            y_start = max(0, y)
            
            face_roi = frame[y_start:y_end, x_start:x_end]
            
            if face_roi.size > 0:
                face_roi_blurred = cv2.GaussianBlur(face_roi, (99, 99), 0)
                frame[y_start:y_end, x_start:x_end] = face_roi_blurred
        
        return frame
    
    def detect_hog(self, frame):
        """DETECTOR 1: HOG + SVM - OTIMIZADO"""
        detections = []
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        boxes, weights = self.hog.detectMultiScale(
            small_frame, winStride=(8, 8), padding=(4, 4), 
            scale=1.05, hitThreshold=0.5
        )
        
        for (x, y, w, h), weight in zip(boxes, weights):
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            # FOCO NA CABEÇA
            head_height = h // 4
            head_y = y
            head_w = w // 2
            head_x = x + (w - head_w) // 2
            head_cx = head_x + head_w // 2
            head_cy = head_y + head_height // 2
            
            detections.append({
                'bbox': (x, y, x+w, y+h),
                'head_bbox': (head_x, head_y, head_x+head_w, head_y+head_height),
                'centroid': (head_cx, head_cy),
                'confidence': float(weight)
            })
        
        return detections
    
    def detect_yolo(self, frame):
        """DETECTOR 2: YOLOv8n - OTIMIZADO"""
        h, w = frame.shape[:2]
        target_size = 640
        scale = target_size / max(h, w)
        
        if scale < 1.0:
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            frame_small = frame
            scale = 1.0
        
        results = self.yolo(frame_small, classes=[0], conf=0.5, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                body_width = x2 - x1
                body_height = y2 - y1
                head_height = body_height * 0.25
                head_width = body_width * 0.5
                
                head_x1 = x1 + (body_width - head_width) / 2
                head_y1 = y1
                head_x2 = head_x1 + head_width
                head_y2 = head_y1 + head_height
                
                head_cx = int((head_x1 + head_x2) / 2)
                head_cy = int((head_y1 + head_y2) / 2)
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'head_bbox': (int(head_x1), int(head_y1), int(head_x2), int(head_y2)),
                    'centroid': (head_cx, head_cy),
                    'confidence': float(confidence)
                })
        
        return detections
    
    def detect_mediapipe(self, frame):
        """DETECTOR 3: MediaPipe Pose"""
        if self.mp_pose is None:
            return []
        
        detections = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(frame_rgb)
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark
            
            nose = landmarks[self.mp.solutions.pose.PoseLandmark.NOSE]
            left_ear = landmarks[self.mp.solutions.pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[self.mp.solutions.pose.PoseLandmark.RIGHT_EAR]
            left_eye = landmarks[self.mp.solutions.pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[self.mp.solutions.pose.PoseLandmark.RIGHT_EYE]
            
            if (nose.visibility > 0.5 and 
                (left_ear.visibility > 0.3 or right_ear.visibility > 0.3)):
                
                head_cx = int(nose.x * w)
                head_cy = int(nose.y * h)
                
                face_points_x = []
                face_points_y = []
                
                for landmark in [nose, left_ear, right_ear, left_eye, right_eye]:
                    if landmark.visibility > 0.3:
                        face_points_x.append(landmark.x * w)
                        face_points_y.append(landmark.y * h)
                
                if face_points_x and face_points_y:
                    head_x1 = max(0, int(min(face_points_x)) - 20)
                    head_y1 = max(0, int(min(face_points_y)) - 20)
                    head_x2 = min(w, int(max(face_points_x)) + 20)
                    head_y2 = min(h, int(max(face_points_y)) + 40)
                    
                    body_x1 = head_x1 - 30
                    body_y1 = head_y1
                    body_x2 = head_x2 + 30
                    body_y2 = min(h, head_y2 + 150)
                    
                    detections.append({
                        'bbox': (body_x1, body_y1, body_x2, body_y2),
                        'head_bbox': (head_x1, head_y1, head_x2, head_y2),
                        'centroid': (head_cx, head_cy),
                        'confidence': nose.visibility,
                        'landmarks': results.pose_landmarks
                    })
        
        return detections
    
    def check_crossing(self, current_centroids, previous_centroids, 
                       recent_crossings, line_x, frame_number, detector_name):
        """Verifica cruzamentos ESQUERDA → DIREITA"""
        crossings = 0
        
        if len(previous_centroids) == 0:
            return crossings, current_centroids
        
        for curr_cx, curr_cy in current_centroids:
            min_dist = float('inf')
            best_prev = None
            
            for prev_cx, prev_cy in previous_centroids:
                dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                if dist < min_dist and dist < self.max_move:
                    min_dist = dist
                    best_prev = (prev_cx, prev_cy)
            
            if best_prev is None:
                continue
            
            prev_cx, prev_cy = best_prev
            crossed = (prev_cx < line_x and curr_cx >= line_x)
            
            if crossed:
                position_key = f"{int(curr_cx)//50}_{int(curr_cy)//50}_{frame_number//5}"
                
                if position_key not in recent_crossings:
                    crossings += 1
                    recent_crossings.append(position_key)
        
        return crossings, current_centroids
    
    def draw_detections(self, frame, detections_dict, line_x):
        """Desenha os 3 detectores em cores diferentes"""
        height, width = frame.shape[:2]
        
        # Linha vertical amarela
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
        cv2.putText(frame, "LINHA", (line_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # HOG (Vermelho)
        for det in detections_dict.get('hog', []):
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "HOG", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            if 'head_bbox' in det:
                hx1, hy1, hx2, hy2 = det['head_bbox']
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
        
        # YOLO (Verde)
        for det in detections_dict.get('yolo', []):
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "YOLO", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if 'head_bbox' in det:
                hx1, hy1, hx2, hy2 = det['head_bbox']
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 3)
        
        # MediaPipe (Azul)
        for det in detections_dict.get('mediapipe', []):
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "MP", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            if 'head_bbox' in det:
                hx1, hy1, hx2, hy2 = det['head_bbox']
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 3)
            
            if 'landmarks' in det and det['landmarks']:
                self.mp_drawing.draw_landmarks(
                    frame, det['landmarks'],
                    self.mp.solutions.pose.POSE_CONNECTIONS
                )
        
        # Painel de informações
        info_bg = np.zeros((150, width, 3), dtype=np.uint8)
        
        cv2.putText(info_bg, "COMPARACAO 3 DETECTORES + BLUR:", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bg, f"HOG+SVM: {self.count_hog}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(info_bg, f"YOLOv8n: {self.count_yolo}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(info_bg, f"MediaPipe: {self.count_mediapipe}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        frame = np.vstack([info_bg, frame])
        return frame
    
    def select_line_position(self, frame):
        """Interface para definir linha vertical"""
        print("\n🎯 DEFININDO LINHA DE CONTAGEM:")
        print("   📌 Clique para posicionar | ENTER para confirmar\n")
        
        line_x = int(frame.shape[1] * 0.5)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal line_x
            if event == cv2.EVENT_LBUTTONDOWN:
                line_x = x
                print(f"   📍 Linha movida para X = {x}")
        
        window_name = 'Definir Linha'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            frame_copy = frame.copy()
            cv2.line(frame_copy, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 3)
            cv2.putText(frame_copy, "CLIQUE = mover | ENTER = confirmar", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"X = {line_x}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame_copy)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 13 or key == 27:  # ENTER ou ESC
                break
        
        cv2.destroyWindow(window_name)
        print(f"   ✅ Linha confirmada em X = {line_x}\n")
        return line_x
    
    def process_video(self, video_path, output_path=None, show_video=True,
                     skip_frames=None, define_line_manually=True, orientation_fix=None,
                     enable_anonymization=True, max_duration_minutes=10, auto_skip=True):
        """
        Processa vídeo comparando 3 detectores - OTIMIZADO
        
        Args:
            skip_frames: Frames a pular (None = automático)
            max_duration_minutes: Limite em minutos (padrão: 10)
            auto_skip: Se True, analisa velocidade automaticamente
            enable_anonymization: Se True, aplica blur em faces
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Erro ao abrir: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orientation_fix in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            width, height = height, width
        
        # Limitar duração
        max_frames = fps * 60 * max_duration_minutes
        video_duration_min = total_frames / fps / 60
        
        if total_frames > max_frames:
            print(f"⚠️  Vídeo tem {video_duration_min:.1f} minutos")
            print(f"🔪 Limitando a {max_duration_minutes} minutos")
            total_frames = int(max_frames)
        
        # Skip adaptativo
        if auto_skip and skip_frames is None:
            skip_frames, scenario_type = self.analyze_scene_velocity(cap)
        elif skip_frames is None:
            skip_frames = 60
            scenario_type = "configuração manual"
        
        frames_to_process = total_frames // skip_frames
        estimated_time_min = (frames_to_process * 0.4) / 60  # ~0.4s por frame com 3 detectores
        
        print(f"\n{'='*60}")
        print(f"📹 VÍDEO: {video_path}")
        print(f"{'='*60}")
        print(f"📊 Resolução: {width}x{height} @ {fps}fps")
        print(f"⏱️  Duração total do vídeo: {video_duration_min:.1f} minutos")
        print(f"🔪 Processando: {max_duration_minutes} minutos ({total_frames} frames)")
        print(f"⚡ Skip: {skip_frames} frames ({scenario_type})")
        print(f"🎯 Frames a processar: ~{frames_to_process}")
        print(f"🔒 Blur de faces: {'SIM' if enable_anonymization else 'NÃO'}")
        print(f"⏱️  Tempo estimado: ~{estimated_time_min:.1f} minutos")
        print(f"{'='*60}\n")
        
        # Definir linha
        line_x = int(width * self.line_position)
        if define_line_manually:
            ret, first_frame = cap.read()
            if ret:
                if orientation_fix is not None:
                    first_frame = cv2.rotate(first_frame, orientation_fix)
                line_x = self.select_line_position(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps // skip_frames,
                                 (width, height + 150))
        
        frame_number = 0
        frames_processed = 0
        
        print("🎬 PROCESSAMENTO INICIADO!")
        print(f"🎯 Linha em X = {line_x}")
        print(f"📊 Progresso será mostrado a cada 5 segundos\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_number >= max_frames:
                    break
                
                if orientation_fix is not None:
                    frame = cv2.rotate(frame, orientation_fix)
                
                frame_number += 1
                
                # ⚡ CRÍTICO: Skip ANTES de qualquer processamento
                if frame_number % skip_frames != 0:
                    continue
                
                frames_processed += 1
                
                # ⚡ OTIMIZAÇÃO: Anonimizar SÓ frames processados
                if enable_anonymization:
                    frame_processed = self.anonymize_faces(frame.copy())
                else:
                    frame_processed = frame.copy()
                
                # Detectar com os 3 métodos
                det_hog = self.detect_hog(frame_processed)
                det_yolo = self.detect_yolo(frame_processed)
                det_mediapipe = self.detect_mediapipe(frame_processed)
                
                # Extrair centroides das cabeças
                centroids_hog = [d['centroid'] for d in det_hog]
                centroids_yolo = [d['centroid'] for d in det_yolo]
                centroids_mediapipe = [d['centroid'] for d in det_mediapipe]
                
                # Verificar cruzamentos
                cross_hog, self.prev_centroids_hog = self.check_crossing(
                    centroids_hog, self.prev_centroids_hog,
                    self.recent_crossings_hog, line_x, frame_number, "HOG")
                
                cross_yolo, self.prev_centroids_yolo = self.check_crossing(
                    centroids_yolo, self.prev_centroids_yolo,
                    self.recent_crossings_yolo, line_x, frame_number, "YOLO")
                
                cross_mp, self.prev_centroids_mediapipe = self.check_crossing(
                    centroids_mediapipe, self.prev_centroids_mediapipe,
                    self.recent_crossings_mediapipe, line_x, frame_number, "MediaPipe")
                
                # Atualizar contadores
                self.count_hog += cross_hog
                self.count_yolo += cross_yolo
                self.count_mediapipe += cross_mp
                self.frame_count = frame_number
                
                # Mostrar contagens
                if cross_hog > 0:
                    print(f"🔴 [HOG] Frame {frame_number} → Total: {self.count_hog}")
                if cross_yolo > 0:
                    print(f"🟢 [YOLO] Frame {frame_number} → Total: {self.count_yolo}")
                if cross_mp > 0:
                    print(f"🔵 [MediaPipe] Frame {frame_number} → Total: {self.count_mediapipe}")
                
                # Log
                self.detections_log.append({
                    'frame': frame_number,
                    'timestamp': frame_number / fps,
                    'hog': {'detections': len(det_hog), 'crossings': cross_hog},
                    'yolo': {'detections': len(det_yolo), 'crossings': cross_yolo},
                    'mediapipe': {'detections': len(det_mediapipe), 'crossings': cross_mp}
                })
                
                # Desenhar
                detections_dict = {
                    'hog': det_hog,
                    'yolo': det_yolo,
                    'mediapipe': det_mediapipe
                }
                frame_display = self.draw_detections(frame_processed.copy(), detections_dict, line_x)
                
                if out:
                    out.write(frame_display)
                
                # Progresso a cada 5 segundos
                if frame_number % (fps * 5) == 0:
                    progress = (frame_number / total_frames) * 100
                    elapsed_min = frame_number / fps / 60
                    remaining_min = ((total_frames - frame_number) / fps / 60) * (frames_processed / frame_number)
                    print(f"\n📊 Progresso: {progress:.1f}% | {elapsed_min:.1f}/{total_frames/fps/60:.1f} min")
                    print(f"   🎯 CONTADOS → HOG: {self.count_hog} | YOLO: {self.count_yolo} | MP: {self.count_mediapipe}")
                    print(f"   📈 Frames: {frame_number}/{total_frames} | Processados: {frames_processed}")
                    print(f"   ⏰ Tempo restante: ~{remaining_min:.1f} min\n")
                
                # Mostrar vídeo (opcional)
                if show_video:
                    cv2.imshow('Comparacao 3 Detectores', frame_display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Interrompido pelo usuário")
                        break
        
        except KeyboardInterrupt:
            print(f"\n⏹️  INTERROMPIDO pelo usuário no frame {frame_number}")
            print(f"📊 Frames processados: {frames_processed}")
        finally:
            cap.release()
            if out:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # Resultados finais
        duration_processed = frame_number / fps / 60
        print(f"\n{'='*60}")
        print(f"🏁 RESULTADOS FINAIS - ESQUERDA → DIREITA")
        print(f"{'='*60}")
        print(f"⏱️  Tempo processado: {duration_processed:.1f} minutos")
        print(f"📊 Frames analisados: {frame_number}")
        print(f"🎯 HOG + SVM:     {self.count_hog}")
        print(f"🎯 YOLOv8n:       {self.count_yolo}")
        print(f"🎯 MediaPipe:     {self.count_mediapipe}")
        if duration_processed >= max_duration_minutes:
            print(f"⚠️  Processamento limitado a {max_duration_minutes} minutos")
        print(f"{'='*60}\n")
        
        return {
            'hog': self.count_hog,
            'yolo': self.count_yolo,
            'mediapipe': self.count_mediapipe,
            'frames_processed': frame_number,
            'duration_minutes': duration_processed,
            'scenario_type': scenario_type if auto_skip else "manual"
        }
    
    def save_results(self, filename='resultados_comparacao_3_detectores.json'):
        """Salva resultados comparativos"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'hog_svm': self.count_hog,
                'yolov8n': self.count_yolo,
                'mediapipe_pose': self.count_mediapipe,
                'frames_processados': self.frame_count
            },
            'detections_log': self.detections_log
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados salvos em: {filename}")


# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 COMPARAÇÃO DE 3 DETECTORES - VERSÃO OTIMIZADA")
    print("="*60)
    print("✅ HOG + SVM (Vermelho)")
    print("✅ YOLOv8n (Verde)")
    print("✅ MediaPipe Pose (Azul)")
    print("✅ Skip adaptativo baseado na velocidade")
    print("✅ Anonimização de faces")
    print("="*60 + "\n")
    
    print("📂 Selecione o vídeo...")
    Tk().withdraw()
    VIDEO_PATH = filedialog.askopenfilename(
        title="Selecione o vídeo",
        filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
    )
    
    if not VIDEO_PATH:
        print("❌ Nenhum vídeo selecionado")
        exit()
    
    print(f"✅ Vídeo selecionado: {VIDEO_PATH}\n")
    
    # Criar contador com 3 detectores
    counter = ThreeDetectorComparison(
        line_position=0.5,
        threshold=30,
        memory_frames=10
    )
    
    # Processar com configurações otimizadas
    results = counter.process_video(
        video_path=VIDEO_PATH,
        output_path="comparacao_3_detectores_otimizado.mp4",
        show_video=True,              # Mostrar visualização
        skip_frames=None,              # None = automático (análise de velocidade)
        define_line_manually=True,     # Interface para definir linha
        orientation_fix=cv2.ROTATE_90_CLOCKWISE,  # Rotação do vídeo
        enable_anonymization=True,     # Blur em faces
        max_duration_minutes= 30,       # ⚡ LIMITE: 10 minutos
        auto_skip=True                 # ⚡ Análise automática de velocidade
    )
    
    # Salvar resultados JSON
    counter.save_results('resultados_comparacao_3_detectores.json')
    
    print("\n🎉 PROCESSAMENTO CONCLUÍDO!")
    print(f"📊 Resultados: {results}")
