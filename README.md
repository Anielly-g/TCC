# 📊 Monitor de Fila - Restaurante Universitário

Sistema de monitoramento de filas em tempo real usando ESP32-CAM e YOLO para detecção de pessoas.

## 🔧 Tecnologias

- **Frontend**: Next.js 14, React 18, TypeScript, TailwindCSS
- **Backend**: Python, YOLO, OpenCV
- **Hardware**: ESP32-CAM

## 🚀 Como executar

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install ultralytics opencv-python requests numpy
python tcc.py
```

## 📱 Funcionalidades

- ✅ Contagem de pessoas em tempo real
- ✅ Tempo médio de espera
- ✅ Interface responsiva
- ✅ Atualização automática a cada 3s
- ✅ Integração com ESP32-CAM

## 🎯 Objetivo

Monitoramento inteligente de filas do restaurante universitário do CEFET-MG para otimizar a experiência dos usuários.