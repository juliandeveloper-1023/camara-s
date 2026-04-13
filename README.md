# 🧠 FaceInsight AI — Análisis Facial Inteligente

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/DeepFace-0.0.89+-00D2FF?style=for-the-badge" alt="DeepFace">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

## 📋 Descripción del Proyecto

**FaceInsight AI** es una aplicación web profesional de **reconocimiento facial** basada en **Machine Learning** que permite a los usuarios subir una imagen y obtener un análisis facial completo en tiempo real.

### ¿Qué analiza?

| Análisis | Descripción | Modelo |
|----------|-------------|--------|
| 🎭 **Emoción** | Detecta 7 emociones: feliz, triste, enojado, sorprendido, miedo, asco, neutral | CNN entrenada con FER-2013 |
| 👤 **Género** | Clasifica entre hombre y mujer | CNN binaria |
| 🎂 **Edad** | Estima la edad aparente del sujeto | Red de regresión |

### Características principales

- ✅ Interfaz moderna con diseño **Glassmorphism**
- ✅ Análisis de imágenes estáticas (sin cámara en tiempo real)
- ✅ Resultados organizados en **tarjetas visuales**
- ✅ Distribución detallada de emociones con porcentajes
- ✅ Imagen anotada con bounding box descargable
- ✅ Indicadores de confianza para cada predicción
- ✅ Manejo robusto de errores
- ✅ Diseño responsive y animaciones suaves

---

## 🧬 ¿Cómo funciona DeepFace?

### Concepto general

**DeepFace** es un framework de Python para análisis y reconocimiento facial que utiliza modelos de **Deep Learning (Aprendizaje Profundo)** preentrenados. Fue desarrollado por Serengil y encapsula múltiples modelos de estado del arte en una interfaz simple y unificada.

### Flujo de procesamiento interno

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Imagen    │───▶│  Detección   │───▶│  Alineación │───▶│   Análisis   │
│  de entrada │    │  de rostro   │    │   facial    │    │  con CNN     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                         │                    │                    │
                   OpenCV/MTCNN/       Normalización de      Modelos CNN
                   RetinaFace          ojos, nariz, boca     especializados
                                                                  │
                                                    ┌─────────────┼─────────────┐
                                                    ▼             ▼             ▼
                                              ┌──────────┐ ┌──────────┐ ┌──────────┐
                                              │ Emoción  │ │  Género  │ │   Edad   │
                                              │ (7 cls)  │ │ (binario)│ │(regresión│
                                              └──────────┘ └──────────┘ └──────────┘
```

### Explicación detallada de cada etapa

#### 1. Detección de rostros
DeepFace utiliza un **detector facial** (en este proyecto: OpenCV Haar Cascade) para localizar los rostros dentro de la imagen. El detector identifica las coordenadas `(x, y, ancho, alto)` del bounding box de cada rostro.

#### 2. Alineación facial
Una vez detectado el rostro, se **alinea** normalizando la posición de los puntos faciales clave (ojos, nariz, boca). Esta alineación mejora significativamente la precisión del análisis posterior.

#### 3. Análisis con Redes Neuronales Convolucionales (CNN)

- **Modelo de Emociones**: Red neuronal convolucional entrenada con el dataset **FER-2013** (35,887 imágenes de rostros etiquetados con 7 emociones). Clasifica entre: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

- **Modelo de Género**: CNN binaria entrenada con datasets demográficos faciales. Clasifica entre `Man` (Hombre) y `Woman` (Mujer), proporcionando un porcentaje de confianza.

- **Modelo de Edad**: Red de regresión que predice la **edad aparente** del sujeto como un valor numérico entero.

### ¿Por qué usar DeepFace?

| Ventaja | Descripción |
|---------|-------------|
| 🎯 **Precisión** | Utiliza modelos de estado del arte preentrenados con millones de imágenes |
| 🚀 **Facilidad** | API simple: una sola línea de código para análisis completo |
| 🧠 **Modelos preentrenados** | No requiere GPU costosa ni datasets masivos para entrenar |
| 🔄 **Mantenimiento activo** | Biblioteca con actualizaciones constantes y comunidad activa |
| 🔧 **Versatilidad** | Soporta múltiples backends de detección y modelos de reconocimiento |

---

## 📁 Estructura del Proyecto

```
/facial_app
│
├── app.py                      # Aplicación principal de Streamlit
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación (este archivo)
│
├── utils/
│   ├── __init__.py             # Inicializador del paquete
│   ├── analysis.py             # Lógica de análisis facial con DeepFace
│   └── image_processing.py     # Procesamiento y manipulación de imágenes
│
└── assets/
    └── styles.css              # Estilos CSS personalizados (Glassmorphism)
```

### Responsabilidad de cada archivo

| Archivo | Responsabilidad |
|---------|----------------|
| `app.py` | Orquesta la interfaz, maneja eventos del usuario, renderiza componentes visuales |
| `utils/analysis.py` | Encapsula DeepFace, traduce resultados al español, maneja errores de detección |
| `utils/image_processing.py` | Carga, redimensiona, anota y convierte imágenes entre formatos |
| `assets/styles.css` | Define el diseño Glassmorphism, animaciones, colores y personalización de Streamlit |

---

## 📚 Librerías Utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **Streamlit** | ≥1.28.0 | Framework web para crear la interfaz de usuario interactiva |
| **DeepFace** | ≥0.0.89 | Análisis facial: emoción, género y edad con modelos CNN |
| **OpenCV** | ≥4.8.0 | Procesamiento de imágenes, dibujo de anotaciones y detección facial |
| **NumPy** | ≥1.24.0 | Representación de imágenes como arrays numéricos multidimensionales |
| **Pillow** | ≥10.0.0 | Lectura de archivos JPG/PNG y conversión de formatos de imagen |
| **TensorFlow/Keras** | ≥2.15.0 | Backend de Deep Learning para ejecutar los modelos CNN de DeepFace |

---

## 🚀 Instrucciones para Ejecutar el Proyecto

### Prerrequisitos

- **Python 3.9** o superior instalado
- **pip** actualizado (`pip install --upgrade pip`)
- Conexión a internet (para descargar modelos en la primera ejecución)

### Paso 1: Clonar o descargar el proyecto

```bash
# Si tienes el repositorio en Git:
git clone <URL_DEL_REPOSITORIO>
cd facial_app

# O simplemente navega a la carpeta del proyecto:
cd facial_app
```

### Paso 2: Crear un entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows:
venv\Scripts\activate

# Activar en macOS/Linux:
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** La primera instalación puede tardar varios minutos debido a TensorFlow y sus dependencias.

### Paso 4: Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### Paso 5: Usar la aplicación

1. Haz clic en el botón de subir imagen o arrastra un archivo JPG/PNG.
2. Presiona el botón **"🔍 Analizar Rostro"**.
3. Espera a que la IA procese la imagen.
4. Revisa los resultados en las tarjetas de emoción, género y edad.
5. Descarga la imagen anotada si lo deseas.

---

## ☁️ Instrucciones para Desplegar en Streamlit Cloud

### Paso 1: Subir el proyecto a GitHub

```bash
# Inicializar repositorio Git
git init
git add .
git commit -m "Proyecto de reconocimiento facial con DeepFace"

# Conectar con repositorio remoto
git remote add origin https://github.com/TU_USUARIO/facial-analysis-app.git
git push -u origin main
```

### Paso 2: Configurar Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io).
2. Inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"**.
4. Selecciona tu repositorio, rama (`main`) y archivo principal (`app.py`).
5. Haz clic en **"Deploy"**.

### Paso 3: Configuración avanzada (opcional)

Crea un archivo `.streamlit/config.toml` en la raíz del proyecto:

```toml
[theme]
primaryColor = "#6C63FF"
backgroundColor = "#0F0A1A"
secondaryBackgroundColor = "#1A1033"
textColor = "#E2E8F0"
font = "sans serif"

[server]
maxUploadSize = 200
```

### Notas sobre el despliegue

- ⚠️ **Primera carga:** Los modelos de DeepFace se descargan automáticamente en la primera ejecución (~100-500 MB). Esto puede tardar unos minutos.
- ⚠️ **Recursos:** Streamlit Cloud ofrece recursos limitados. Para imágenes muy grandes, la aplicación puede ser lenta.
- ✅ **Actualizaciones:** Cada `push` a GitHub actualiza automáticamente la aplicación desplegada.

---

## 🎨 Diseño de la Interfaz

La interfaz utiliza un diseño **Glassmorphism** con las siguientes características:

- **Fondo oscuro** con degradado (`#0F0A1A → #1A1033 → #0D1B2A`)
- **Tarjetas translúcidas** con `backdrop-filter: blur(20px)`
- **Bordes sutiles** con `rgba(255,255,255,0.08)`
- **Degradados vibrantes** en azules, morados y cian
- **Animaciones suaves** (fadeIn, slideUp, float, spin)
- **Tipografía Inter** de Google Fonts
- **Efectos glow** decorativos en el fondo

---

## 📝 Licencia

Proyecto académico desarrollado para fines educativos.

---

<p align="center">
  Desarrollado con ❤️ usando <strong>Streamlit</strong> + <strong>DeepFace</strong> + <strong>OpenCV</strong>
</p>
