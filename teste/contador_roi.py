import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
from tkinter import Tk, filedialog

# =============================
# FACE ANONYMIZATION
# =============================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except:
    _MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe não disponível")


# =============================
# TRACKER POR IoU
# =============================
class PersonTracker:
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def calculate_iou(self, b1, b2):
        x1, y1, x2, y2 = b1
        x1p, y1p, x2p, y2p = b2

        xi1 = max(x1, x1p)
        yi1 = max(y1, y1p)
        xi2 = min(x2, x2p)
        yi2 = min(y2, y2p)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2p - x1p) * (y2p - y1p)
        return inter / (area1 + area2 - inter)

    def update(self, detections):
        matched_dets = set()

        for tid in list(self.tracks.keys()):
            best_iou, best_idx = 0, -1
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self.calculate_iou(self.tracks[tid]['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_idx = iou, i

            if best_idx >= 0:
                # Atualizar histórico de posições
                if 'position_history' not in self.tracks[tid]:
                    self.tracks[tid]['position_history'] = []
                
                self.tracks[tid]['position_history'].append(detections[best_idx]['centroid'][0])
                
                # Manter apenas últimas 10 posições
                if len(self.tracks[tid]['position_history']) > 10:
                    self.tracks[tid]['position_history'] = self.tracks[tid]['position_history'][-10:]
                
                self.tracks[tid].update({
                    'bbox': detections[best_idx]['bbox'],
                    'centroid': detections[best_idx]['centroid'],
                    'age': 0,
                    'hits': self.tracks[tid]['hits'] + 1,
                    'last': detections[best_idx]
                })
                matched_dets.add(best_idx)
            else:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'centroid': det['centroid'],
                    'age': 0,
                    'hits': 1,
                    'crossed': False,
                    'last': det,
                    'position_history': [det['centroid'][0]]  # Histórico de posições X
                }
                self.next_id += 1

        confirmed = []
        for tid, t in self.tracks.items():
            if t['hits'] >= self.min_hits:
                confirmed.append({
                    'id': tid,
                    'bbox': t['bbox'],
                    'centroid': t['centroid'],
                    'crossed': t['crossed'],
                    'detection': t['last']
                })
        return confirmed

    def mark_crossed(self, tid):
        if tid in self.tracks:
            self.tracks[tid]['crossed'] = True


# =============================
# SISTEMA PRINCIPAL
# =============================
class ThreeDetectorWithROI:
    def __init__(self):
        print("🔧 Carregando detectores...")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.yolo = YOLO("yolov8n.pt")

        if _MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose.Pose()
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.mp_pose = None

        self.tracker_hog = PersonTracker()
        self.tracker_yolo = PersonTracker()
        self.tracker_mp = PersonTracker()

        self.count_hog = 0
        self.count_yolo = 0
        self.count_mp = 0

        self.roi = None
        self.roi_selected = False
        self.selecting_roi = False
        self.roi_start_point = None
        self.roi_end_point = None
        
        # Variáveis para linha de contagem móvel
        self.line_x = None
        self.dragging_line = False
        self.line_tolerance = 15  # Tolerância para clicar na linha
        
        # Controle de estado
        self.setup_complete = False
        self.paused = True
        
        # Informações de vídeo para progresso
        self.total_frames = 0
        self.video_fps = 0
        self.start_time = None
        
        # Coleta de dados para relatório
        self.detection_log = []
        self.video_name = ""
        self.roi_coordinates = None
        self.line_position = None
        
        # Controle de progressão no terminal
        self.last_progress_print = 0
        self.progress_interval = 5  # Imprimir a cada 5%
        
        # Limitação de tempo de análise
        self.max_analysis_minutes = 30
        self.max_analysis_seconds = self.max_analysis_minutes * 60  # 1800 segundos
        
        # Configurações de display
        self.max_window_width = 1200
        self.max_window_height = 800
        self.scale_factor = 1.0
        self.display_frame_size = None

    # -------------------------
    def anonymize(self, frame):
        if FACE_CASCADE.empty():
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 0)
        return frame

    # -------------------------
    def resize_frame_for_display(self, frame):
        """Redimensiona frame para caber na tela mantendo proporção"""
        height, width = frame.shape[:2]
        
        # Calcular fator de escala se necessário
        scale_w = self.max_window_width / width if width > self.max_window_width else 1.0
        scale_h = self.max_window_height / height if height > self.max_window_height else 1.0
        scale = min(scale_w, scale_h, 1.0)  # Só reduzir, nunca ampliar
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Salvar fator de escala para converter coordenadas do mouse
            self.scale_factor = scale
            self.display_frame_size = (new_width, new_height)
            
            return resized_frame
        else:
            self.scale_factor = 1.0
            self.display_frame_size = (width, height)
            return frame.copy()
    
    # -------------------------
    def convert_mouse_coords(self, x, y):
        """Converte coordenadas do mouse da janela redimensionada para coordenadas originais"""
        if self.scale_factor < 1.0:
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            return orig_x, orig_y
        return x, y
    
    # -------------------------
    def convert_coords_to_display(self, x, y):
        """Converte coordenadas originais para coordenadas da janela redimensionada"""
        if self.scale_factor < 1.0:
            display_x = int(x * self.scale_factor)
            display_y = int(y * self.scale_factor)
            return display_x, display_y
        return x, y

    # -------------------------
    def mouse_callback(self, event, x, y, flags, param):
        # Converter coordenadas da janela redimensionada para originais
        orig_x, orig_y = self.convert_mouse_coords(x, y)
        
        # Verificar se está clicando próximo à linha de contagem
        if self.line_x and abs(orig_x - self.line_x) <= self.line_tolerance:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging_line = True
                print(f"📏 Iniciando arraste da linha em x = {self.line_x}")
                return
                
        if event == cv2.EVENT_LBUTTONDOWN and not self.dragging_line:
            self.selecting_roi = True
            self.roi_start_point = (orig_x, orig_y)
            self.roi_end_point = (orig_x, orig_y)
            print(f"📏 Iniciando seleção ROI em ({orig_x}, {orig_y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_line:
                # Arrastar linha de contagem
                self.line_x = orig_x
                # Não fazer print a cada movimento para evitar spam
            elif self.selecting_roi:
                # Arrastar ROI
                self.roi_end_point = (orig_x, orig_y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_line:
                self.dragging_line = False
                print(f"✅ Linha posicionada em x = {self.line_x}")
            elif self.selecting_roi:
                self.roi_end_point = (orig_x, orig_y)
                self.selecting_roi = False
                
                # Definir ROI
                x1 = min(self.roi_start_point[0], self.roi_end_point[0])
                y1 = min(self.roi_start_point[1], self.roi_end_point[1])
                x2 = max(self.roi_start_point[0], self.roi_end_point[0])
                y2 = max(self.roi_start_point[1], self.roi_end_point[1])
                
                if x2 - x1 > 20 and y2 - y1 > 20:  # ROI mínimo
                    self.roi = (x1, y1, x2, y2)
                    self.roi_selected = True
                    print(f"🎯 ROI definido: ({x1}, {y1}, {x2}, {y2})")
                    print(f"⚙️ Pressione ESPAÇO para iniciar processamento")

    # -------------------------
    def point_in_roi(self, point):
        """Verifica se um ponto está dentro do ROI"""
        if not self.roi:
            return True  # Se não há ROI, aceita todos os pontos
        
        x, y = point
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2

    # -------------------------
    def filter_detections_by_roi(self, detections):
        """Filtra detecções que estão dentro do ROI"""
        if not self.roi:
            return detections
        
        filtered = []
        for det in detections:
            # Verificar se o centroid está dentro do ROI
            if self.point_in_roi(det['centroid']):
                filtered.append(det)
        return filtered

    # -------------------------
    def resize_frame_for_display(self, frame):
        """Redimensiona frame para caber na tela mantendo proporção"""
        height, width = frame.shape[:2]
        
        # Calcular fator de escala se necessário
        scale_w = self.max_window_width / width if width > self.max_window_width else 1.0
        scale_h = self.max_window_height / height if height > self.max_window_height else 1.0
        scale = min(scale_w, scale_h, 1.0)  # Só reduzir, nunca ampliar
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Salvar fator de escala para converter coordenadas do mouse
            self.scale_factor = scale
            self.display_frame_size = (new_width, new_height)
            
            return resized_frame
        else:
            self.scale_factor = 1.0
            self.display_frame_size = (width, height)
            return frame.copy()
    
    # -------------------------
    def convert_mouse_coords(self, x, y):
        """Converte coordenadas do mouse da janela redimensionada para coordenadas originais"""
        if self.scale_factor < 1.0:
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            return orig_x, orig_y
        return x, y
    
    # -------------------------
    def convert_coords_to_display(self, x, y):
        """Converte coordenadas originais para coordenadas da janela redimensionada"""
        if self.scale_factor < 1.0:
            display_x = int(x * self.scale_factor)
            display_y = int(y * self.scale_factor)
            return display_x, display_y
        return x, y

    # -------------------------
    def format_time(self, seconds):
        """Converte segundos em formato MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    # -------------------------
    def draw_progress_bar(self, frame, current_frame):
        """Desenha barra de progresso e informações de tempo"""
        if self.total_frames == 0:
            return
        
        # Calcular progresso
        progress = current_frame / self.total_frames
        
        # Tempo atual e total do vídeo
        current_time_video = current_frame / self.video_fps
        total_time_video = self.total_frames / self.video_fps
        
        # Tempo de processamento
        if self.start_time:
            processing_time = time.time() - self.start_time
            estimated_total_processing = processing_time / progress if progress > 0 else 0
            remaining_processing = max(0, estimated_total_processing - processing_time)
        else:
            processing_time = 0
            remaining_processing = 0
        
        # Dimensões da barra
        bar_width = frame.shape[1] - 40
        bar_height = 20
        bar_x = 20
        bar_y = frame.shape[0] - 100
        
        # Fundo da barra
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
        
        # Barra de progresso
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Texto de progresso
        progress_text = f"{progress * 100:.1f}%"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y + (bar_height + text_size[1]) // 2
        cv2.putText(frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Informações de tempo
        time_info = (
            f"Vídeo: {self.format_time(current_time_video)} / {self.format_time(total_time_video)} | "
            f"Restante: {self.format_time(total_time_video - current_time_video)} | "
            f"Processamento: ~{self.format_time(remaining_processing)}"
        )
        
        cv2.putText(frame, time_info, (20, frame.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame info
        frame_info = f"Frame: {current_frame:,} / {self.total_frames:,}"
        cv2.putText(frame, frame_info, (20, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # -------------------------
    def print_progress(self, current_frame):
        """Imprime progresso no terminal"""
        if self.total_frames == 0:
            return
        
        progress = (current_frame / self.total_frames) * 100
        
        # Só imprimir a cada 5% para não poluir o terminal
        if progress - self.last_progress_print >= self.progress_interval:
            self.last_progress_print = progress
            
            # Tempo atual e estimativa
            current_time_video = current_frame / self.video_fps
            total_time_video = self.total_frames / self.video_fps
            remaining_video = total_time_video - current_time_video
            
            # Tempo de processamento
            if self.start_time:
                processing_time = time.time() - self.start_time
                if progress > 0:
                    estimated_total = processing_time * 100 / progress
                    remaining_processing = max(0, estimated_total - processing_time)
                else:
                    remaining_processing = 0
            else:
                processing_time = 0
                remaining_processing = 0
            
            # Status de contagem
            total_detections = self.count_hog + self.count_yolo + self.count_mp
            
            print(f"\r📊 [{progress:5.1f}%] {self.format_time(current_time_video)}/{self.format_time(total_time_video)} | "
                  f"Restante: {self.format_time(remaining_video)} (~{self.format_time(remaining_processing)} proc.) | "
                  f"Contagem: H:{self.count_hog} Y:{self.count_yolo} M:{self.count_mp} (Total:{total_detections})", 
                  end='', flush=True)

    # -------------------------
    def log_detection_data(self, frame_num, det_hog, det_yolo, det_mp):
        """Registra dados de detecção para relatório"""
        timestamp = frame_num / self.video_fps if self.video_fps > 0 else 0
        
        log_entry = {
            "frame": frame_num,
            "timestamp": round(timestamp, 2),
            "hog": {
                "detections": len(det_hog),
                "crossings": self.count_hog
            },
            "yolo": {
                "detections": len(det_yolo),
                "crossings": self.count_yolo
            },
            "mediapipe": {
                "detections": len(det_mp),
                "crossings": self.count_mp
            }
        }
        
        self.detection_log.append(log_entry)
    
    # -------------------------
    def save_results(self):
        """Salva resultados completos em arquivo JSON"""
        import json
        import os
        from datetime import datetime
        
        # Calcular estatísticas
        processing_time = time.time() - self.start_time if self.start_time else 0
        frames_processados = len(self.detection_log)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "name": self.video_name,
                "total_frames": self.total_frames,
                "fps": self.video_fps,
                "duration_seconds": self.total_frames / self.video_fps if self.video_fps > 0 else 0,
                "frames_processados": frames_processados
            },
            "setup": {
                "roi_coordinates": self.roi_coordinates,
                "line_position": self.line_position
            },
            "performance": {
                "processing_time_seconds": round(processing_time, 2),
                "fps_processing": round(frames_processados / processing_time, 2) if processing_time > 0 else 0,
                "skip_rate": 30  # Processando 1 a cada 30 frames
            },
            "summary": {
                "hog_svm": self.count_hog,
                "yolov8n": self.count_yolo,
                "mediapipe_pose": self.count_mp
            },
            "detection_stats": {
                "total_hog_detections": sum(log['hog']['detections'] for log in self.detection_log),
                "total_yolo_detections": sum(log['yolo']['detections'] for log in self.detection_log),
                "total_mp_detections": sum(log['mediapipe']['detections'] for log in self.detection_log)
            },
            "detections_log": self.detection_log
        }
        
        # Nome do arquivo de saída
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_base = os.path.splitext(os.path.basename(self.video_name))[0] if self.video_name else "video"
        filename = f"resultados_{video_base}_{timestamp}.json"
        
        # Salvar arquivo
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📊 RESULTADOS SALVOS: {filename}")
            print(f"\n" + "="*50)
            print(f"📊 RESUMO FINAL DA ANÁLISE")
            print(f"="*50)
            print(f"   🎬 Vídeo: {os.path.basename(self.video_name)}")
            print(f"   ⏱️ Duração: {self.format_time(self.total_frames / self.video_fps if self.video_fps > 0 else 0)}")
            print(f"   🖥️ Frames processados: {frames_processados:,} / {self.total_frames:,}")
            print(f"   ⏱️ Tempo processamento: {self.format_time(processing_time)}")
            print(f"")
            print(f"   📊 COMPARAÇÃO DE DETECTORES:")
            print(f"   • 🔴 HOG+SVM: {self.count_hog} pessoas")
            print(f"   • 🟢 YOLOv8n: {self.count_yolo} pessoas")
            print(f"   • 🔵 MediaPipe: {self.count_mp} pessoas")
            print(f"")
            if self.count_yolo > 0:
                hog_vs_yolo = (self.count_hog / self.count_yolo) * 100
                mp_vs_yolo = (self.count_mp / self.count_yolo) * 100
                print(f"   📊 PERFORMANCE RELATIVA (vs YOLO):")
                print(f"   • HOG: {hog_vs_yolo:.1f}% das detecções do YOLO")
                print(f"   • MediaPipe: {mp_vs_yolo:.1f}% das detecções do YOLO")
            print(f"="*50)
            
        except Exception as e:
            print(f"\n❌ Erro ao salvar resultados: {e}")

    # -------------------------
    def detect_hog(self, frame):
        dets = []
        
        # Se há ROI, processar apenas essa região
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
            
            if roi_frame.size > 0:
                boxes, _ = self.hog.detectMultiScale(roi_frame, winStride=(8, 8))
                for (x, y, w, h) in boxes:
                    # Ajustar coordenadas para o frame completo
                    abs_x = x + x1
                    abs_y = y + y1
                    abs_x2 = abs_x + w
                    abs_y2 = abs_y + h
                    
                    cx, cy = abs_x + w // 2, abs_y + h // 4
                    dets.append({
                        'bbox': (abs_x, abs_y, abs_x2, abs_y2),
                        'centroid': (cx, cy)
                    })
        else:
            # Se não há ROI, processar frame completo
            boxes, _ = self.hog.detectMultiScale(frame, winStride=(8, 8))
            for (x, y, w, h) in boxes:
                cx, cy = x + w // 2, y + h // 4
                dets.append({
                    'bbox': (x, y, x+w, y+h),
                    'centroid': (cx, cy)
                })
        return dets

    # -------------------------
    def detect_yolo(self, frame):
        dets = []
        
        # Se há ROI, processar apenas essa região
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
            
            if roi_frame.size > 0:
                res = self.yolo(roi_frame, classes=[0], conf=0.5, verbose=False)
                for r in res:
                    for b in r.boxes:
                        x, y, x_end, y_end = map(int, b.xyxy[0])
                        
                        # Ajustar coordenadas para o frame completo
                        abs_x1 = x + x1
                        abs_y1 = y + y1
                        abs_x2 = x_end + x1
                        abs_y2 = y_end + y1
                        
                        cx, cy = (abs_x1 + abs_x2) // 2, abs_y1 + (abs_y2 - abs_y1) // 4
                        dets.append({
                            'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                            'centroid': (cx, cy)
                        })
        else:
            # Se não há ROI, processar frame completo
            res = self.yolo(frame, classes=[0], conf=0.5, verbose=False)
            for r in res:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cx, cy = (x1+x2)//2, y1 + (y2-y1)//4
                    dets.append({
                        'bbox': (x1, y1, x2, y2),
                        'centroid': (cx, cy)
                    })
        return dets

    # -------------------------
    def detect_mediapipe(self, frame):
        dets = []
        if not self.mp_pose:
            return dets
        
        # Se há ROI, processar apenas essa região
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
            
            if roi_frame.size > 0:
                rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                results = self.mp_pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    h, w = roi_frame.shape[:2]
                    landmarks = results.pose_landmarks.landmark
                    
                    # Usar pontos do torso para criar bounding box
                    shoulder_left = landmarks[11]  # LEFT_SHOULDER
                    shoulder_right = landmarks[12] # RIGHT_SHOULDER
                    hip_left = landmarks[23]       # LEFT_HIP
                    hip_right = landmarks[24]      # RIGHT_HIP
                    
                    # Calcular bounding box baseado nos landmarks do torso
                    x_coords = [shoulder_left.x, shoulder_right.x, hip_left.x, hip_right.x]
                    y_coords = [shoulder_left.y, shoulder_right.y, hip_left.y, hip_right.y]
                    
                    x_min = int(min(x_coords) * w) - 50  # Margem
                    x_max = int(max(x_coords) * w) + 50
                    y_min = int(min(y_coords) * h) - 50
                    y_max = int(max(y_coords) * h) + 100  # Mais margem para baixo
                    
                    # Garantir que está dentro dos limites da ROI
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    if x_max > x_min and y_max > y_min:
                        # Ajustar coordenadas para o frame completo
                        abs_x_min = x_min + x1
                        abs_y_min = y_min + y1
                        abs_x_max = x_max + x1
                        abs_y_max = y_max + y1
                        
                        cx = (abs_x_min + abs_x_max) // 2
                        cy = abs_y_min + (abs_y_max - abs_y_min) // 4
                        
                        dets.append({
                            'bbox': (abs_x_min, abs_y_min, abs_x_max, abs_y_max),
                            'centroid': (cx, cy)
                        })
        else:
            # Se não há ROI, processar frame completo
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_frame)
            
            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.pose_landmarks.landmark
                
                # Usar pontos do torso para criar bounding box
                shoulder_left = landmarks[11]  # LEFT_SHOULDER
                shoulder_right = landmarks[12] # RIGHT_SHOULDER
                hip_left = landmarks[23]       # LEFT_HIP
                hip_right = landmarks[24]      # RIGHT_HIP
                
                # Calcular bounding box baseado nos landmarks do torso
                x_coords = [shoulder_left.x, shoulder_right.x, hip_left.x, hip_right.x]
                y_coords = [shoulder_left.y, shoulder_right.y, hip_left.y, hip_right.y]
                
                x_min = int(min(x_coords) * w) - 50  # Margem
                x_max = int(max(x_coords) * w) + 50
                y_min = int(min(y_coords) * h) - 50
                y_max = int(max(y_coords) * h) + 100  # Mais margem para baixo
                
                # Garantir que está dentro dos limites da imagem
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                if x_max > x_min and y_max > y_min:
                    cx = (x_min + x_max) // 2
                    cy = y_min + (y_max - y_min) // 4  # Centro superior para tracking
                    
                    dets.append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'centroid': (cx, cy)
                    })
                    
        return dets

    # -------------------------
    def check_cross(self, tracks, line_x, tracker, counter):
        """Detecta travessias reais da esquerda para direita"""
        for t in tracks:
            if not t['crossed']:
                track_id = t['id']
                
                # Verificar se temos histórico de posições suficiente
                if track_id in tracker.tracks and 'position_history' in tracker.tracks[track_id]:
                    history = tracker.tracks[track_id]['position_history']
                    
                    if len(history) >= 3:  # Precisamos de pelo menos 3 pontos
                        # Verificar se houve travessia
                        recent_positions = history[-3:]  # Últimas 3 posições
                        
                        # Pessoa estava à esquerda e agora está à direita?
                        was_left = any(pos < line_x - 10 for pos in recent_positions[:-1])  # Margem de 10px
                        is_right = recent_positions[-1] > line_x + 10
                        
                        if was_left and is_right:
                            tracker.mark_crossed(track_id)
                            counter += 1
                            print(f"\n✅ TRAVESSIA DETECTADA! ID: {track_id} | Total: {counter}")
                            print(f"   • Posições recentes: {recent_positions}")
                            print(f"   • Linha em x={line_x}\n")
                            
        return counter

    # =============================
    def process_video(self, video_path):
        self.video_name = video_path  # Salvar nome do vídeo
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Erro ao abrir vídeo")
            return
        
        # Obter informações do vídeo
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        print(f"🎥 Vídeo carregado:")
        print(f"   • Frames: {self.total_frames:,}")
        print(f"   • FPS: {self.video_fps:.1f}")
        print(f"   • Duração total: {self.format_time(total_duration)}")
        
        # Calcular limitação
        max_frames = int(self.max_analysis_seconds * self.video_fps) if self.video_fps > 0 else self.total_frames
        analysis_duration = min(total_duration, self.max_analysis_seconds)
        
        print(f"   • ⏱️ ANÁLISE LIMITADA: primeiros {self.max_analysis_minutes} minutos ({self.format_time(analysis_duration)})")
        if total_duration > self.max_analysis_seconds:
            print(f"   • ⚠️ Vídeo será interrompido em {self.format_time(self.max_analysis_seconds)}")

        ret, frame = cap.read()
        if not ret:
            return
        
        # Aplicar rotação ao primeiro frame para definir dimensões
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        self.line_x = frame.shape[1] // 2  # Posição inicial da linha
        
        print(f"📏 Dimensões originais: {frame.shape[1]}x{frame.shape[0]}")
        
        # Configurar janela redimensionável
        cv2.namedWindow("Contador", cv2.WINDOW_NORMAL)
        
        # Calcular tamanho da janela
        display_frame = self.resize_frame_for_display(frame)
        window_width, window_height = self.display_frame_size
        
        print(f"📺 Janela redimensionada: {window_width}x{window_height} (escala: {self.scale_factor:.2f})")
        
        # Definir tamanho da janela
        cv2.resizeWindow("Contador", window_width, window_height)
        cv2.setMouseCallback("Contador", self.mouse_callback)
        
        print("🎯 Configure o sistema antes de iniciar:")
        print("1. Arraste o mouse para definir ROI")
        print("2. Clique na linha amarela para reposicioná-la")
        print("3. Pressione ESPAÇO para iniciar ou 'q' para sair")
        print(f"⏰ ANÁLISE LIMITADA: Primeiros {self.max_analysis_minutes} minutos apenas")
        
        # Loop de configuração inicial
        while self.paused:
            # Mostrar frame de configuração redimensionado
            config_frame = self.resize_frame_for_display(frame.copy())
            
            # Converter coordenadas para visualização
            display_line_x = int(self.line_x * self.scale_factor) if self.line_x else 0
            
            # Desenhar linha de contagem
            line_color = (0, 255, 255)  # Amarelo padrão
            line_thickness = 3
            
            if self.dragging_line:
                line_color = (0, 255, 0)  # Verde quando arrastando
                line_thickness = 5
            
            cv2.line(config_frame, (display_line_x, 0), (display_line_x, config_frame.shape[0]), line_color, line_thickness)
            cv2.circle(config_frame, (display_line_x, int(50 * self.scale_factor)), int(10 * self.scale_factor), line_color, -1)
            cv2.putText(config_frame, "LINHA", (display_line_x - int(30 * self.scale_factor), int(35 * self.scale_factor)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale_factor, line_color, max(1, int(2 * self.scale_factor)))
            
            # Desenhar ROI
            if self.roi:
                x1, y1, x2, y2 = self.roi
                # Converter para coordenadas da tela
                dx1, dy1 = self.convert_coords_to_display(x1, y1)
                dx2, dy2 = self.convert_coords_to_display(x2, y2)
                
                cv2.rectangle(config_frame, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)
                cv2.putText(config_frame, "ROI", (dx1, dy1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale_factor, (255, 255, 0), max(1, int(2 * self.scale_factor)))
            elif self.selecting_roi and self.roi_start_point and self.roi_end_point:
                # Desenhar ROI sendo selecionado
                x1 = min(self.roi_start_point[0], self.roi_end_point[0])
                y1 = min(self.roi_start_point[1], self.roi_end_point[1])
                x2 = max(self.roi_start_point[0], self.roi_end_point[0])
                y2 = max(self.roi_start_point[1], self.roi_end_point[1])
                
                # Converter para coordenadas da tela
                dx1, dy1 = self.convert_coords_to_display(x1, y1)
                dx2, dy2 = self.convert_coords_to_display(x2, y2)
                
                cv2.rectangle(config_frame, (dx1, dy1), (dx2, dy2), (255, 255, 255), 2)
            
            # Instruções de configuração
            cv2.putText(config_frame, "CONFIGURACAO INICIAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if not self.roi_selected:
                cv2.putText(config_frame, "1. Arraste mouse para definir ROI", (10, config_frame.shape[0]-70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(config_frame, "2. Clique na linha para reposicionar", (10, config_frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(config_frame, "Pressione 'q' para sair", (10, config_frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(config_frame, "ROI OK! Pressione ESPACO para iniciar", (10, config_frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(config_frame, "'r' = redefinir | 'q' = sair", (10, config_frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Contador", config_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):  # Redefinir ROI
                self.roi = None
                self.roi_selected = False
                print("🔄 ROI resetado")
            elif key == ord(' ') and self.roi_selected:  # Espaço para iniciar
                self.paused = False
                self.setup_complete = True
                self.start_time = time.time()  # Iniciar cronômetro
                
                # Salvar configurações
                self.roi_coordinates = self.roi
                self.line_position = self.line_x
                
                print("▶️ Iniciando processamento!")
                print(f"🎯 ROI: {self.roi}")
                print(f"📏 Linha: x={self.line_x}")
                print(f"🔄 Atualizando progresso a cada {self.progress_interval}%...\n")
        
        # Configurar callback do mouse para seleção do ROI
        cv2.namedWindow("Contador")
        cv2.setMouseCallback("Contador", self.mouse_callback)

        frame_idx = 0
        print("🎉 Sistema configurado! Processando vídeo...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar rotação a todos os frames
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Verificar limite de tempo (30 minutos)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time_seconds = current_frame / self.video_fps if self.video_fps > 0 else 0
            
            if current_time_seconds >= self.max_analysis_seconds:
                print(f"\n⏰ LIMITE DE TEMPO ATINGIDO: {self.max_analysis_minutes} minutos")
                print(f"📊 Interrompendo análise em {self.format_time(current_time_seconds)}")
                break

            frame_idx += 1
            if frame_idx % 30 != 0:
                continue

            # Executar todos os três detectores separadamente
            det_hog = self.detect_hog(frame)
            det_yolo = self.detect_yolo(frame)
            det_mp = self.detect_mediapipe(frame)

            # Atualizar trackers separados
            tracks_hog = self.tracker_hog.update(det_hog)
            tracks_yolo = self.tracker_yolo.update(det_yolo)
            tracks_mp = self.tracker_mp.update(det_mp)

            # Verificar cruzamentos separados
            self.count_hog = self.check_cross(
                tracks_hog, self.line_x, self.tracker_hog, self.count_hog
            )
            self.count_yolo = self.check_cross(
                tracks_yolo, self.line_x, self.tracker_yolo, self.count_yolo
            )
            self.count_mp = self.check_cross(
                tracks_mp, self.line_x, self.tracker_mp, self.count_mp
            )

            frame = self.anonymize(frame)

            # Desenhar linha de cruzamento com estilo interativo
            line_color = (0, 255, 255)  # Amarelo padrão
            line_thickness = 3
            
            if self.dragging_line:
                line_color = (0, 255, 0)  # Verde quando arrastando
                line_thickness = 5
            
            cv2.line(frame, (self.line_x, 0), (self.line_x, frame.shape[0]), line_color, line_thickness)
            
            # Adicionar indicador visual para arraste
            cv2.circle(frame, (self.line_x, 50), 10, line_color, -1)
            cv2.putText(frame, "LINHA", (self.line_x - 30, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
            
            # Desenhar ROI
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "ROI", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif self.selecting_roi and self.roi_start_point and self.roi_end_point:
                # Desenhar ROI sendo selecionado
                x1 = min(self.roi_start_point[0], self.roi_end_point[0])
                y1 = min(self.roi_start_point[1], self.roi_end_point[1])
                x2 = max(self.roi_start_point[0], self.roi_end_point[0])
                y2 = max(self.roi_start_point[1], self.roi_end_point[1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Instruções
            if not self.roi_selected:
                cv2.putText(frame, "Arraste o mouse para definir ROI", (10, frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Pressione 'r' para redefinir | 'q' para sair", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "ROI Definido! Pressione 'r' para redefinir", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Desenhar bounding boxes dos detectores separados
            
            # HOG - Vermelho  
            for t in tracks_hog:
                x1, y1, x2, y2 = t['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Mostrar direção do movimento
                track_id = t['id']
                direction = ""
                if track_id in self.tracker_hog.tracks and 'position_history' in self.tracker_hog.tracks[track_id]:
                    history = self.tracker_hog.tracks[track_id]['position_history']
                    if len(history) >= 2:
                        if history[-1] > history[-2]:
                            direction = " →"
                        elif history[-1] < history[-2]:
                            direction = " ←"
                
                cv2.putText(frame, f"HOG-{t['id']}{direction}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # YOLO - Verde
            for t in tracks_yolo:
                x1, y1, x2, y2 = t['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Mostrar direção do movimento
                track_id = t['id']
                direction = ""
                if track_id in self.tracker_yolo.tracks and 'position_history' in self.tracker_yolo.tracks[track_id]:
                    history = self.tracker_yolo.tracks[track_id]['position_history']
                    if len(history) >= 2:
                        if history[-1] > history[-2]:
                            direction = " →"
                        elif history[-1] < history[-2]:
                            direction = " ←"
                
                cv2.putText(frame, f"YOLO-{t['id']}{direction}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # MediaPipe - Azul
            for t in tracks_mp:
                x1, y1, x2, y2 = t['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Mostrar direção do movimento  
                track_id = t['id']
                direction = ""
                if track_id in self.tracker_mp.tracks and 'position_history' in self.tracker_mp.tracks[track_id]:
                    history = self.tracker_mp.tracks[track_id]['position_history']
                    if len(history) >= 2:
                        if history[-1] > history[-2]:
                            direction = " →"
                        elif history[-1] < history[-2]:
                            direction = " ←"
                
                cv2.putText(frame, f"MP-{t['id']}{direction}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Desenhar detecções individuais (mais fracas, para análise)
            alpha = 0.3  # Transparência
            alpha = 0.3  # Transparência
            
            # HOG - Vermelho (sutil)
            for t in tracks_hog:
                x1, y1, x2, y2 = t['bbox']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # YOLO - Verde (sutil)
            for t in tracks_yolo:
                x1, y1, x2, y2 = t['bbox']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # MediaPipe - Azul (sutil)
            for t in tracks_mp:
                x1, y1, x2, y2 = t['bbox']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Contadores separados - cada detector independente
            cv2.putText(frame, f"HOG+SVM: {self.count_hog}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"YOLO: {self.count_yolo}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"MediaPipe: {self.count_mp}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Obter frame atual para progresso
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Imprimir progresso no terminal
            self.print_progress(current_frame)
            
            # Registrar dados para relatório
            self.log_detection_data(current_frame, det_hog, det_yolo, det_mp)
            
            # Contadores individuais (para análise)
            cv2.putText(frame, f"HOG: {self.count_hog} | YOLO: {self.count_yolo} | MP: {self.count_mp}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Barra de progresso e informações de tempo
            self.draw_progress_bar(frame, current_frame)
            
            # Instruções
            if not self.roi_selected:
                cv2.putText(frame, "Arraste mouse: ROI | Clique linha: mover contagem", (10, frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Pressione 'r' para redefinir | 'q' para sair", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "ROI OK! Clique na linha para reposicionar contagem", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Redimensionar frame para exibição
            display_frame = self.resize_frame_for_display(frame)

            cv2.imshow("Contador", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Redefinir ROI
                self.roi = None
                self.roi_selected = False
                print("🔄 ROI resetado - selecione nova região")

        cap.release()
        cv2.destroyAllWindows()
        
        # Mensagem de conclusão
        print("\n\n✅ PROCESSAMENTO CONCLUÍDO!")
        
        # Salvar resultados
        if self.detection_log:  # Só salvar se houver dados
            self.save_results()


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    Tk().withdraw()
    video = filedialog.askopenfilename(
        title="Selecione o vídeo",
        filetypes=[("Vídeos", "*.mp4 *.avi *.mov")]
    )

    if video:
        system = ThreeDetectorWithROI()
        system.process_video(video)
