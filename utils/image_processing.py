# ============================================================
# MÓDULO DE PROCESAMIENTO DE IMÁGENES - image_processing.py
# ============================================================
#
# DESCRIPCIÓN:
#   Este módulo contiene funciones auxiliares para el preprocesamiento
#   y manipulación de imágenes antes y después del análisis facial.
#   Maneja la conversión de formatos, redimensionamiento y generación
#   de la imagen con anotaciones visuales del resultado.
#
# RESPONSABILIDAD:
#   - Convertir archivos cargados por Streamlit a arrays NumPy
#   - Redimensionar imágenes para optimizar el procesamiento
#   - Dibujar bounding boxes y anotaciones sobre los rostros detectados
#   - Convertir imágenes entre espacios de color (RGB ↔ BGR)
#
# ============================================================

import cv2
import numpy as np
from PIL import Image
import io


def cargar_imagen(archivo_subido) -> np.ndarray:
    """
    Convierte un archivo subido en Streamlit a un array NumPy (RGB).

    FUNCIONAMIENTO:
        1. Lee los bytes del archivo subido por el usuario.
        2. Usa PIL (Pillow) para interpretar los bytes como imagen.
        3. Convierte la imagen al espacio de color RGB (Pillow usa RGB por defecto).
        4. Convierte la imagen PIL a un array NumPy para compatibilidad con
           OpenCV y DeepFace.

    PARÁMETROS:
        archivo_subido: Objeto UploadedFile de Streamlit que contiene
                        la imagen en formato JPG o PNG.

    RETORNA:
        np.ndarray: Imagen como array NumPy en formato RGB con dimensiones
                    (alto, ancho, canales).

    NOTA:
        DeepFace acepta arrays NumPy directamente, lo cual evita la
        necesidad de guardar la imagen temporalmente en disco.
    """
    # Leer la imagen usando PIL desde los bytes del archivo
    imagen_pil = Image.open(archivo_subido)

    # Asegurar que la imagen esté en modo RGB
    # (algunas imágenes pueden venir en RGBA, L, CMYK, etc.)
    if imagen_pil.mode != "RGB":
        imagen_pil = imagen_pil.convert("RGB")

    # Convertir la imagen PIL a array NumPy
    imagen_array = np.array(imagen_pil)

    return imagen_array


def redimensionar_imagen(imagen: np.ndarray, max_ancho: int = 800) -> np.ndarray:
    """
    Redimensiona una imagen manteniendo la proporción si excede el ancho máximo.

    FUNCIONAMIENTO:
        Si la imagen supera el ancho máximo especificado, se calcula un factor
        de escala proporcional y se redimensiona usando interpolación bicúbica
        (cv2.INTER_AREA para reducir, que produce mejores resultados al
        disminuir el tamaño).

    RESPONSABILIDAD:
        Optimizar el procesamiento reduciendo el tamaño de imágenes muy grandes
        sin perder calidad perceptible para el análisis facial.

    PARÁMETROS:
        imagen (np.ndarray): Imagen original como array NumPy.
        max_ancho (int): Ancho máximo permitido en píxeles (por defecto: 800).

    RETORNA:
        np.ndarray: Imagen redimensionada o la imagen original si no excede
                    el ancho máximo.
    """
    alto, ancho = imagen.shape[:2]

    if ancho > max_ancho:
        # Calcular el factor de escala proporcional
        factor = max_ancho / ancho
        nuevo_ancho = max_ancho
        nuevo_alto = int(alto * factor)

        # Redimensionar con interpolación de área (ideal para reducir tamaño)
        imagen_redimensionada = cv2.resize(
            imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
        )
        return imagen_redimensionada

    return imagen


def dibujar_anotaciones(
    imagen: np.ndarray, resultado: dict
) -> np.ndarray:
    """
    Dibuja el bounding box y etiquetas sobre el rostro detectado.

    FUNCIONAMIENTO:
        1. Extrae las coordenadas del rostro (x, y, ancho, alto) del resultado.
        2. Dibuja un rectángulo con bordes redondeados alrededor del rostro.
        3. Agrega etiquetas con la emoción, género y edad sobre el rectángulo.
        4. Aplica un fondo semitransparente detrás de las etiquetas para
           mejorar la legibilidad.

    RESPONSABILIDAD:
        Generar una versión anotada de la imagen original para que el usuario
        pueda visualizar exactamente dónde se detectó el rostro y los
        resultados del análisis directamente sobre la imagen.

    PARÁMETROS:
        imagen (np.ndarray): Imagen original como array NumPy (RGB).
        resultado (dict): Diccionario con los resultados del análisis facial.

    RETORNA:
        np.ndarray: Imagen con las anotaciones dibujadas (RGB).
    """
    # Crear una copia de la imagen para no modificar la original
    imagen_anotada = imagen.copy()

    # Obtener las coordenadas del rostro detectado
    region = resultado.get("rostro_region", {})
    x = region.get("x", 0)
    y = region.get("y", 0)
    w = region.get("w", 0)
    h = region.get("h", 0)

    # Si no hay región válida, devolver la imagen sin cambios
    if w == 0 or h == 0:
        return imagen_anotada

    # --- Convertir de RGB a BGR para usar funciones de OpenCV ---
    # OpenCV trabaja internamente con BGR, no RGB
    imagen_bgr = cv2.cvtColor(imagen_anotada, cv2.COLOR_RGB2BGR)

    # --- Color del borde: púrpura vibrante (#6C63FF) ---
    color_borde = (255, 99, 108)  # BGR para #6C63FF
    grosor_borde = 2

    # --- Dibujar esquinas redondeadas del bounding box ---
    # En lugar de un rectángulo simple, dibujamos esquinas estilizadas
    longitud_esquina = min(w, h) // 4

    # Esquina superior izquierda
    cv2.line(imagen_bgr, (x, y), (x + longitud_esquina, y), color_borde, grosor_borde)
    cv2.line(imagen_bgr, (x, y), (x, y + longitud_esquina), color_borde, grosor_borde)

    # Esquina superior derecha
    cv2.line(imagen_bgr, (x + w, y), (x + w - longitud_esquina, y), color_borde, grosor_borde)
    cv2.line(imagen_bgr, (x + w, y), (x + w, y + longitud_esquina), color_borde, grosor_borde)

    # Esquina inferior izquierda
    cv2.line(imagen_bgr, (x, y + h), (x + longitud_esquina, y + h), color_borde, grosor_borde)
    cv2.line(imagen_bgr, (x, y + h), (x, y + h - longitud_esquina), color_borde, grosor_borde)

    # Esquina inferior derecha
    cv2.line(imagen_bgr, (x + w, y + h), (x + w - longitud_esquina, y + h), color_borde, grosor_borde)
    cv2.line(imagen_bgr, (x + w, y + h), (x + w, y + h - longitud_esquina), color_borde, grosor_borde)

    # --- Preparar texto de la etiqueta ---
    from utils.analysis import EMOCIONES_NOMBRE

    emocion_clave = resultado.get("emocion_clave", "neutral")
    emocion_nombre = EMOCIONES_NOMBRE.get(emocion_clave, emocion_clave)
    edad = resultado.get("edad", 0)
    genero_clave = resultado.get("genero_clave", "")
    genero_texto = "H" if genero_clave == "Man" else "M"

    etiqueta = f"{emocion_nombre} | {genero_texto} | {edad} años"

    # --- Dibujar fondo semitransparente para la etiqueta ---
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala_fuente = 0.55
    grosor_texto = 1

    # Calcular el tamaño del texto para dimensionar el fondo
    (ancho_texto, alto_texto), linea_base = cv2.getTextSize(
        etiqueta, fuente, escala_fuente, grosor_texto
    )

    # Posición de la etiqueta (encima del bounding box)
    padding = 8
    pos_y_etiqueta = max(y - 10, alto_texto + padding + 5)

    # Dibujar rectángulo de fondo (semitransparente simulado con rectángulo relleno)
    overlay = imagen_bgr.copy()
    cv2.rectangle(
        overlay,
        (x, pos_y_etiqueta - alto_texto - padding),
        (x + ancho_texto + padding * 2, pos_y_etiqueta + padding // 2),
        (108, 99, 255),  # BGR para fondo púrpura
        -1,  # Relleno completo
    )
    # Aplicar transparencia con addWeighted
    alpha = 0.85
    imagen_bgr = cv2.addWeighted(overlay, alpha, imagen_bgr, 1 - alpha, 0)

    # Dibujar el texto sobre el fondo
    cv2.putText(
        imagen_bgr,
        etiqueta,
        (x + padding, pos_y_etiqueta - padding // 2),
        fuente,
        escala_fuente,
        (255, 255, 255),  # Texto blanco
        grosor_texto,
        cv2.LINE_AA,  # Antialiasing para texto suave
    )

    # --- Convertir de vuelta a RGB ---
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    return imagen_rgb


def imagen_a_bytes(imagen: np.ndarray, formato: str = "PNG") -> bytes:
    """
    Convierte un array NumPy de imagen a bytes para descarga.

    RESPONSABILIDAD:
        Permitir que el usuario descargue la imagen anotada con los
        resultados del análisis facial.

    PARÁMETROS:
        imagen (np.ndarray): Imagen como array NumPy (RGB).
        formato (str): Formato de salida ('PNG' o 'JPEG').

    RETORNA:
        bytes: Imagen codificada como bytes.
    """
    imagen_pil = Image.fromarray(imagen)
    buffer = io.BytesIO()
    imagen_pil.save(buffer, format=formato)
    buffer.seek(0)
    return buffer.getvalue()
