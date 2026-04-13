# ============================================================
# MÓDULO DE ANÁLISIS FACIAL - analysis.py
# ============================================================
#
# DESCRIPCIÓN:
#   Este módulo encapsula toda la lógica de análisis facial
#   utilizando la librería DeepFace. Proporciona funciones
#   para detectar emociones, estimar género y predecir edad
#   a partir de una imagen cargada por el usuario.
#
# ¿QUÉ ES DEEPFACE?
#   DeepFace es una librería de Python desarrollada por Serengil
#   que actúa como un framework ligero para el reconocimiento y
#   análisis facial. Internamente utiliza modelos de Deep Learning
#   preentrenados (redes neuronales convolucionales - CNN) que han
#   sido entrenados con millones de imágenes faciales.
#
# ¿CÓMO FUNCIONA DEEPFACE A NIVEL CONCEPTUAL?
#   1. DETECCIÓN DE ROSTROS: Primero localiza los rostros dentro
#      de la imagen usando detectores como OpenCV, RetinaFace, MTCNN, etc.
#   2. ALINEACIÓN: Alinea el rostro detectado para normalizar la posición
#      de los ojos, nariz y boca, mejorando la precisión del análisis.
#   3. REPRESENTACIÓN: Convierte el rostro en un vector numérico (embedding)
#      usando modelos como VGG-Face, FaceNet, ArcFace, entre otros.
#   4. ANÁLISIS DE ATRIBUTOS: Utiliza modelos especializados para predecir:
#      - Emoción: Clasificación con 7 clases (feliz, triste, enojado,
#        sorprendido, miedo, asco, neutral) usando una CNN entrenada
#        con el dataset FER-2013.
#      - Género: Clasificación binaria (hombre/mujer) usando una CNN
#        entrenada con datos demográficos faciales.
#      - Edad: Regresión que estima la edad aparente del sujeto.
#
# ¿POR QUÉ USAR DEEPFACE?
#   - PRECISIÓN: Utiliza modelos de estado del arte preentrenados con
#     millones de imágenes, logrando alta precisión sin necesidad de
#     entrenar modelos propios.
#   - FACILIDAD: API simple de una sola línea de código para obtener
#     análisis completo (emoción, género, edad, raza).
#   - MODELOS PREENTRENADOS: No requiere GPU costosa ni datasets
#     masivos; los modelos ya han sido entrenados por expertos.
#   - MANTENIMIENTO ACTIVO: Biblioteca con actualizaciones constantes
#     y una comunidad activa en GitHub.
#   - VERSATILIDAD: Soporta múltiples backends de detección facial
#     y modelos de reconocimiento, permitiendo ajustar precisión vs velocidad.
#
# ============================================================

from deepface import DeepFace
import numpy as np


# --- Diccionario de traducción de emociones (inglés → español) ---
EMOCIONES_ES = {
    "angry": "Enojado 😠",
    "disgust": "Asco 🤢",
    "fear": "Miedo 😨",
    "happy": "Feliz 😊",
    "sad": "Triste 😢",
    "surprise": "Sorprendido 😲",
    "neutral": "Neutral 😐",
}

# --- Diccionario de traducción de género (inglés → español) ---
GENERO_ES = {
    "Man": "Hombre 👨",
    "Woman": "Mujer 👩",
}

# --- Emojis para cada emoción (sin texto) ---
EMOCIONES_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# --- Nombres cortos de emociones en español (sin emoji) ---
EMOCIONES_NOMBRE = {
    "angry": "Enojado",
    "disgust": "Asco",
    "fear": "Miedo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Sorprendido",
    "neutral": "Neutral",
}


def analizar_rostro(imagen_array: np.ndarray) -> dict:
    """
    Analiza un rostro en una imagen y devuelve emoción, género y edad.

    FUNCIONAMIENTO:
        Esta función utiliza DeepFace.analyze() que internamente:
        1. Detecta todos los rostros presentes en la imagen.
        2. Para cada rostro, ejecuta modelos CNN especializados:
           - Modelo de emociones: CNN entrenada con FER-2013 (35,887 imágenes)
           - Modelo de género: CNN binaria entrenada con datos demográficos
           - Modelo de edad: Red de regresión que predice la edad aparente
        3. Devuelve las predicciones con sus niveles de confianza.

    PARÁMETROS:
        imagen_array (np.ndarray): Imagen en formato array de NumPy (BGR o RGB).

    RETORNA:
        dict: Diccionario con las claves:
            - 'exito' (bool): Indica si el análisis fue exitoso.
            - 'emocion_dominante' (str): Emoción principal detectada en español.
            - 'emocion_emoji' (str): Emoji representativo de la emoción.
            - 'emociones_detalle' (dict): Porcentajes de todas las emociones.
            - 'genero' (str): Género detectado en español con emoji.
            - 'genero_confianza' (float): Nivel de confianza del género (%).
            - 'edad' (int): Edad estimada del sujeto.
            - 'rostro_region' (dict): Coordenadas del rostro detectado.
            - 'error' (str): Mensaje de error si el análisis falla.
    """
    try:
        # =============================================================
        # DeepFace.analyze() - Función principal de análisis
        # =============================================================
        # Parámetros utilizados:
        #   - img_path: La imagen a analizar (acepta ruta, URL o array NumPy).
        #   - actions: Lista de análisis a realizar ['emotion', 'gender', 'age'].
        #     No incluimos 'race' porque no es requisito del proyecto.
        #   - enforce_detection: Si es True, lanza error cuando no detecta rostro.
        #     Lo ponemos en True para manejar el error de forma controlada.
        #   - detector_backend: Algoritmo de detección facial a usar.
        #     'opencv' es rápido y confiable para la mayoría de casos.
        #   - silent: Si es True, suprime los mensajes de log de DeepFace.
        # =============================================================
        resultados = DeepFace.analyze(
            img_path=imagen_array,
            actions=["emotion", "gender", "age"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )

        # DeepFace puede devolver una lista si detecta múltiples rostros.
        # Tomamos el primer rostro detectado para el análisis principal.
        if isinstance(resultados, list):
            resultado = resultados[0]
        else:
            resultado = resultados

        # --- Extraer la emoción dominante ---
        # 'dominant_emotion' contiene la clave en inglés de la emoción
        # con mayor probabilidad según el modelo CNN de emociones.
        emocion_clave = resultado.get("dominant_emotion", "neutral")
        emocion_dominante = EMOCIONES_ES.get(emocion_clave, emocion_clave)
        emocion_emoji = EMOCIONES_EMOJI.get(emocion_clave, "🤔")

        # --- Extraer el detalle de todas las emociones ---
        # 'emotion' contiene un diccionario con los porcentajes de confianza
        # para cada una de las 7 emociones del modelo FER-2013.
        emociones_raw = resultado.get("emotion", {})
        emociones_detalle = {}
        for clave, valor in emociones_raw.items():
            nombre_es = EMOCIONES_NOMBRE.get(clave, clave)
            emociones_detalle[nombre_es] = round(float(valor), 2)

        # --- Extraer el género ---
        # 'dominant_gender' contiene 'Man' o 'Woman'.
        # 'gender' contiene los porcentajes de confianza para cada género.
        genero_clave = resultado.get("dominant_gender", "Man")
        genero = GENERO_ES.get(genero_clave, genero_clave)
        genero_confianza_dict = resultado.get("gender", {})
        genero_confianza = round(float(genero_confianza_dict.get(genero_clave, 0)), 2)

        # --- Extraer la edad estimada ---
        # 'age' contiene un valor numérico entero con la edad aparente.
        edad = int(resultado.get("age", 0))

        # --- Extraer la región del rostro detectado ---
        # 'region' contiene las coordenadas (x, y, w, h) del bounding box.
        rostro_region = resultado.get("region", {})

        return {
            "exito": True,
            "emocion_dominante": emocion_dominante,
            "emocion_emoji": emocion_emoji,
            "emocion_clave": emocion_clave,
            "emociones_detalle": emociones_detalle,
            "genero": genero,
            "genero_clave": genero_clave,
            "genero_confianza": genero_confianza,
            "edad": edad,
            "rostro_region": rostro_region,
            "error": None,
        }

    except ValueError as e:
        # =============================================================
        # Error más común: no se detectó ningún rostro en la imagen.
        # DeepFace lanza ValueError cuando enforce_detection=True y no
        # encuentra un rostro válido en la imagen proporcionada.
        # =============================================================
        return {
            "exito": False,
            "error": (
                "No se detectó ningún rostro en la imagen. "
                "Por favor, sube una foto donde se vea claramente un rostro humano."
            ),
        }

    except Exception as e:
        # =============================================================
        # Captura de errores genéricos: problemas de memoria, formatos
        # de imagen no soportados, errores de modelos, etc.
        # =============================================================
        return {
            "exito": False,
            "error": f"Ocurrió un error durante el análisis: {str(e)}",
        }


def obtener_color_emocion(emocion_clave: str) -> str:
    """
    Devuelve un color hexadecimal asociado a cada emoción.

    RESPONSABILIDAD:
        Proporciona una paleta de colores consistente para representar
        visualmente cada emoción en la interfaz gráfica.

    PARÁMETROS:
        emocion_clave (str): Clave de la emoción en inglés.

    RETORNA:
        str: Código de color hexadecimal.
    """
    colores = {
        "happy": "#4ADE80",      # Verde - alegría
        "sad": "#60A5FA",        # Azul - tristeza
        "angry": "#F87171",      # Rojo - enojo
        "surprise": "#FBBF24",   # Amarillo - sorpresa
        "fear": "#A78BFA",       # Púrpura - miedo
        "disgust": "#34D399",    # Verde lima - asco
        "neutral": "#94A3B8",    # Gris - neutral
    }
    return colores.get(emocion_clave, "#94A3B8")
