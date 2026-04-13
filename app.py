# ============================================================
# APLICACIÓN PRINCIPAL - app.py
# ============================================================
#
# PROYECTO: Reconocimiento Facial con Machine Learning
# TECNOLOGÍAS: Streamlit + DeepFace + OpenCV + NumPy + Pillow
#
# DESCRIPCIÓN:
#   Aplicación web profesional que permite al usuario subir una
#   imagen y obtener un análisis facial completo que incluye:
#   - Detección de emoción dominante (7 clases)
#   - Clasificación de género (hombre/mujer)
#   - Estimación de edad aproximada
#
# FUNCIONAMIENTO DE DEEPFACE (Resumen Conceptual):
#   DeepFace es un framework de Python que encapsula modelos de
#   Deep Learning preentrenados para análisis facial. Utiliza redes
#   neuronales convolucionales (CNN) que han sido entrenadas con
#   millones de imágenes faciales. El flujo interno es:
#
#   Imagen → Detección de Rostro → Alineación → Análisis por CNN →
#   → Predicción de Emoción, Género y Edad
#
#   Cada tarea (emoción, género, edad) usa un modelo CNN específico:
#   - Emociones: Modelo entrenado con FER-2013 (35,887 imágenes)
#   - Género: Modelo binario entrenado con datos demográficos
#   - Edad: Modelo de regresión para estimación de edad aparente
#
# JUSTIFICACIÓN DEL USO DE DEEPFACE:
#   1. Precisión: Modelos de estado del arte con alta precisión.
#   2. Facilidad: Una sola línea de código para análisis completo.
#   3. Modelos preentrenados: No requiere entrenamiento ni GPU costosa.
#   4. Comunidad activa: Mantenida y actualizada constantemente.
#
# NOTA DE COMPATIBILIDAD CON STREAMLIT CLOUD:
#   El error 'removeChild' de React ocurre cuando el navegador
#   modifica el DOM de forma inesperada (ej: emojis que crean
#   nodos de texto extra, o <br> sueltos). Para evitarlo:
#   1. Todos los emojis van dentro de <span> explícitos.
#   2. No usar st.markdown("<br>") — usar st.write("") en su lugar.
#   3. Todo el HTML debe ser auto-contenido en cada st.markdown().
#   4. Envolver todo HTML en un único <div> raíz por llamada.
#
# ============================================================

import streamlit as st
import time
import os
from utils.analysis import analizar_rostro, obtener_color_emocion
from utils.image_processing import (
    cargar_imagen,
    redimensionar_imagen,
    dibujar_anotaciones,
    imagen_a_bytes,
)


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# ============================================================
# Configuramos la página con un ícono, título y layout amplio
# para maximizar el espacio visual de la interfaz.
st.set_page_config(
    page_title="FaceInsight AI — Análisis Facial Inteligente",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CARGA DE ESTILOS CSS PERSONALIZADOS
# ============================================================
def cargar_estilos():
    """
    Carga el archivo CSS personalizado desde la carpeta assets.

    RESPONSABILIDAD:
        Inyectar los estilos de glassmorphism, animaciones y diseño
        moderno en la aplicación Streamlit mediante el componente
        st.markdown con unsafe_allow_html=True.
    """
    ruta_css = os.path.join(os.path.dirname(__file__), "assets", "styles.css")

    if os.path.exists(ruta_css):
        with open(ruta_css, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Si no se encuentra el CSS, usar estilos mínimos en línea
        st.markdown(
            """
            <style>
                .stApp {
                    background: linear-gradient(135deg, #0F0A1A, #1A1033, #0D1B2A);
                    font-family: 'Inter', sans-serif;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


# Cargar los estilos al iniciar
cargar_estilos()


# ============================================================
# FUNCIÓN AUXILIAR: Espaciador seguro
# ============================================================
def espaciador():
    """
    Agrega un espacio vertical usando st.write en vez de <br>.
    st.markdown('<br>') causa errores removeChild en React
    porque el navegador interpreta <br> de forma inconsistente.
    """
    st.write("")


# ============================================================
# ELEMENTOS DECORATIVOS DE FONDO (Glows)
# ============================================================
def renderizar_fondo():
    """
    Renderiza los efectos de brillo decorativo en el fondo.

    RESPONSABILIDAD:
        Crear elementos visuales con efecto de resplandor (glow)
        que aportan profundidad y dinamismo al diseño glassmorphism.
    """
    st.markdown(
        '<div>'
        '<div class="bg-glow bg-glow-1"></div>'
        '<div class="bg-glow bg-glow-2"></div>'
        '<div class="bg-glow bg-glow-3"></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ENCABEZADO HERO DE LA APLICACIÓN
# ============================================================
def renderizar_encabezado():
    """
    Renderiza el encabezado principal con título, subtítulo y badge.

    RESPONSABILIDAD:
        Proporcionar un punto focal visual atractivo que presente
        la aplicación de forma profesional y moderna.
    """
    st.markdown(
        '<div class="hero-header">'
        '<h1><span>&#129504;</span> FaceInsight AI</h1>'
        '<p class="hero-subtitle">'
        'An&aacute;lisis facial inteligente &middot; Emoci&oacute;n &middot; G&eacute;nero &middot; Edad'
        '</p>'
        '<span class="hero-badge"><span>&#9889;</span> Powered by DeepFace &amp; Deep Learning</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ZONA DE CARGA DE IMAGEN
# ============================================================
def renderizar_zona_carga():
    """
    Renderiza la zona visual de carga con instrucciones amigables.

    RESPONSABILIDAD:
        Proporcionar una zona de carga con diseño atractivo y
        mensajes claros que guíen al usuario.
    """
    st.markdown(
        '<div class="upload-zone">'
        '<span class="upload-icon"><span>&#128248;</span></span>'
        '<p class="upload-text">Arrastra o selecciona una imagen para analizar</p>'
        '<p class="upload-hint">Formatos soportados: JPG, JPEG, PNG</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# INDICADOR DE CARGA ANIMADO
# ============================================================
def renderizar_cargando():
    """
    Renderiza un indicador de carga personalizado con animación.

    RESPONSABILIDAD:
        Informar al usuario que el análisis está en proceso con
        una animación suave y mensajes descriptivos.
    """
    st.markdown(
        '<div class="loading-container">'
        '<div class="loading-spinner"></div>'
        '<p class="loading-text"><span>&#128269;</span> Analizando rostro con inteligencia artificial...</p>'
        '<p class="loading-subtext">Los modelos de Deep Learning est&aacute;n procesando tu imagen</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TARJETAS DE RESULTADOS INDIVIDUALES
# ============================================================
def renderizar_tarjeta_resultado(
    tipo: str, icono: str, etiqueta: str, valor: str, detalle: str, confianza: float = None
):
    """
    Renderiza una tarjeta de resultado individual con glassmorphism.

    RESPONSABILIDAD:
        Presentar cada resultado del análisis (emoción, género, edad)
        en una tarjeta visual atractiva con icono, valor y barra
        de confianza opcional.

    PARÁMETROS:
        tipo (str): Clase CSS ('emotion', 'gender', 'age').
        icono (str): Emoji representativo del resultado.
        etiqueta (str): Texto de la etiqueta superior.
        valor (str): Valor principal a mostrar.
        detalle (str): Texto descriptivo secundario.
        confianza (float): Porcentaje de confianza (opcional).
    """
    # Construir barra de confianza si se proporciona
    barra_html = ""
    if confianza is not None:
        barra_html = (
            f'<div class="confidence-bar-container">'
            f'<div class="confidence-bar {tipo}-bar" style="width: {confianza}%;"></div>'
            f'</div>'
            f'<p class="result-detail" style="margin-top: 0.5rem; font-size: 0.8rem;">'
            f'Confianza: {confianza:.1f}%'
            f'</p>'
        )

    # NOTA: Todo el HTML está completo y auto-contenido en una sola llamada.
    # Emojis van dentro de <span> para evitar nodos de texto sueltos.
    st.markdown(
        f'<div class="result-card {tipo}">'
        f'<span class="result-icon"><span>{icono}</span></span>'
        f'<p class="result-label">{etiqueta}</p>'
        f'<p class="result-value">{valor}</p>'
        f'<p class="result-detail">{detalle}</p>'
        f'{barra_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SECCIÓN DE DISTRIBUCIÓN DE EMOCIONES
# ============================================================
def renderizar_detalle_emociones(emociones: dict, emocion_dominante_clave: str):
    """
    Renderiza la tabla detallada de distribución de emociones.

    RESPONSABILIDAD:
        Mostrar el porcentaje de confianza de cada una de las 7
        emociones detectadas, permitiendo al usuario entender la
        distribución completa del análisis emocional.

    PARÁMETROS:
        emociones (dict): Diccionario con emoción → porcentaje.
        emocion_dominante_clave (str): Clave de la emoción dominante.
    """
    # Ordenar emociones de mayor a menor porcentaje
    emociones_ordenadas = sorted(emociones.items(), key=lambda x: x[1], reverse=True)

    # Construir todas las filas de emociones como HTML completo
    filas_html = ""
    max_valor = max(emociones.values()) if emociones else 0
    for nombre, porcentaje in emociones_ordenadas:
        # Resaltar la emoción dominante con un color diferente
        color = "#A78BFA" if porcentaje == max_valor else "#64748B"
        peso = "600" if porcentaje == max_valor else "400"

        filas_html += (
            f'<div class="emotion-row">'
            f'<span class="emotion-name" style="font-weight: {peso};">{nombre}</span>'
            f'<span class="emotion-value" style="color: {color};">{porcentaje:.1f}%</span>'
            f'</div>'
        )

    # NOTA: Todo el HTML está completo y auto-contenido en una sola llamada
    st.markdown(
        f'<div class="detail-section">'
        f'<p class="detail-title"><span>&#128202;</span> Distribuci&oacute;n de Emociones</p>'
        f'{filas_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MENSAJE DE ERROR ESTILIZADO
# ============================================================
def renderizar_error(mensaje: str):
    """
    Muestra un mensaje de error con estilo visual coherente.

    PARÁMETROS:
        mensaje (str): Texto del error a mostrar.
    """
    st.markdown(
        f'<div class="status-message error">'
        f'<span style="font-size: 1.5rem;"><span>&#9888;&#65039;</span></span>'
        f'<div>'
        f'<p style="color: #F87171; font-weight: 600; margin: 0;">No se pudo completar el an&aacute;lisis</p>'
        f'<p style="color: #FDA4AF; margin: 0.3rem 0 0; font-size: 0.9rem;">{mensaje}</p>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MENSAJE DE ÉXITO ESTILIZADO
# ============================================================
def renderizar_exito():
    """
    Muestra un mensaje de éxito después del análisis completo.
    """
    st.markdown(
        '<div class="status-message success">'
        '<span style="font-size: 1.5rem;"><span>&#9989;</span></span>'
        '<div>'
        '<p style="color: #4ADE80; font-weight: 600; margin: 0;">An&aacute;lisis completado con &eacute;xito</p>'
        '<p style="color: #86EFAC; margin: 0.3rem 0 0; font-size: 0.9rem;">Los modelos de IA han procesado la imagen correctamente</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PIE DE PÁGINA
# ============================================================
def renderizar_pie():
    """
    Renderiza el pie de página con créditos y tecnologías.
    """
    st.markdown(
        '<div class="app-footer">'
        '<p>'
        '<span>&#129504;</span> FaceInsight AI &middot; Desarrollado con '
        '<strong>Streamlit</strong> + <strong>DeepFace</strong> + <strong>OpenCV</strong>'
        '</p>'
        '<p style="margin-top: 0.3rem;">'
        'Proyecto acad&eacute;mico &mdash; Reconocimiento Facial con Machine Learning &middot; 2025'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FUNCIÓN PRINCIPAL DE LA APLICACIÓN
# ============================================================
def main():
    """
    Función principal que orquesta toda la aplicación.

    FLUJO PRINCIPAL:
        1. Renderizar fondo decorativo y encabezado.
        2. Mostrar la zona de carga de imagen.
        3. Cuando el usuario sube una imagen:
           a. Mostrar vista previa de la imagen.
           b. Procesar con DeepFace (emoción, género, edad).
           c. Mostrar resultados en tarjetas glassmorphism.
           d. Mostrar distribución detallada de emociones.
           e. Ofrecer descarga de imagen anotada.
        4. Renderizar pie de página.

    NOTA TÉCNICA (Compatibilidad Streamlit Cloud):
        - Emojis SIEMPRE dentro de <span> para evitar nodos de texto sueltos.
        - NUNCA usar st.markdown('<br>') — usar st.write('') en su lugar.
        - Todo HTML auto-contenido con un <div> raíz por llamada.
        - Concatenar strings HTML en vez de usar f-strings multilínea.
    """
    # --- Fondo y encabezado ---
    renderizar_fondo()
    renderizar_encabezado()

    # --- Sección de información ---
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(
            '<div class="glass-card" style="text-align: center; padding: 1.2rem;">'
            '<span style="font-size: 2rem;"><span>&#127917;</span></span>'
            '<p style="color: #A78BFA; font-weight: 600; margin: 0.5rem 0 0.2rem;">7 Emociones</p>'
            '<p style="color: #64748B; font-size: 0.8rem; margin: 0;">Detecci&oacute;n precisa</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_info2:
        st.markdown(
            '<div class="glass-card" style="text-align: center; padding: 1.2rem;">'
            '<span style="font-size: 2rem;"><span>&#128100;</span></span>'
            '<p style="color: #00D2FF; font-weight: 600; margin: 0.5rem 0 0.2rem;">G&eacute;nero &amp; Edad</p>'
            '<p style="color: #64748B; font-size: 0.8rem; margin: 0;">Estimaci&oacute;n inteligente</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_info3:
        st.markdown(
            '<div class="glass-card" style="text-align: center; padding: 1.2rem;">'
            '<span style="font-size: 2rem;"><span>&#9889;</span></span>'
            '<p style="color: #FBBF24; font-weight: 600; margin: 0.5rem 0 0.2rem;">Deep Learning</p>'
            '<p style="color: #64748B; font-size: 0.8rem; margin: 0;">Modelos preentrenados</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    espaciador()

    # ============================================================
    # ZONA DE CARGA DE IMAGEN
    # ============================================================
    renderizar_zona_carga()

    # Componente de Streamlit para subir archivos
    # Permite formatos JPG, JPEG y PNG como se requiere
    archivo_subido = st.file_uploader(
        "Sube tu imagen aquí",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen con un rostro visible para el análisis facial.",
        label_visibility="collapsed",
    )

    # ============================================================
    # PROCESAMIENTO DE LA IMAGEN
    # ============================================================
    if archivo_subido is not None:
        # --- Cargar y preparar la imagen ---
        imagen_original = cargar_imagen(archivo_subido)
        imagen_procesada = redimensionar_imagen(imagen_original)

        espaciador()

        # --- Diseño de dos columnas: imagen + resultados ---
        col_imagen, col_resultados = st.columns([1, 1], gap="large")

        with col_imagen:
            # Mostrar la vista previa de la imagen cargada
            st.markdown(
                '<div class="glass-card">'
                '<p class="detail-title" style="margin-bottom: 1rem;">'
                '<span>&#128444;&#65039;</span> Imagen Cargada'
                '</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.image(
                imagen_procesada,
                caption=archivo_subido.name,
                use_container_width=True,
            )

        with col_resultados:
            # --- Botón para iniciar el análisis ---
            boton_analizar = st.button(
                "Analizar Rostro",
                use_container_width=True,
                type="primary",
            )

            if boton_analizar:
                # Mostrar indicador de carga personalizado
                placeholder_carga = st.empty()
                with placeholder_carga.container():
                    renderizar_cargando()

                # ==============================================
                # Ejecutar el análisis facial con DeepFace
                # ==============================================
                # DeepFace.analyze() internamente:
                # 1. Detecta rostros con el backend seleccionado (OpenCV)
                # 2. Alinea el rostro detectado
                # 3. Pasa el rostro por modelos CNN especializados:
                #    - CNN de emociones (FER-2013)
                #    - CNN de género (clasificación binaria)
                #    - CNN de edad (regresión)
                # 4. Devuelve las predicciones con confianzas
                # ==============================================
                resultado = analizar_rostro(imagen_procesada)

                # Simular un tiempo mínimo de carga para la animación
                time.sleep(1)

                # Limpiar el indicador de carga
                placeholder_carga.empty()

                # ==============================================
                # MOSTRAR RESULTADOS
                # ==============================================
                if resultado["exito"]:
                    # Guardar resultados en session_state para persistencia
                    st.session_state["resultado"] = resultado
                    st.session_state["imagen_procesada"] = imagen_procesada

                else:
                    # Mostrar error si no se detectó un rostro
                    renderizar_error(resultado["error"])

            # --- Mostrar resultados almacenados en session_state ---
            if "resultado" in st.session_state and st.session_state["resultado"]["exito"]:
                resultado = st.session_state["resultado"]
                imagen_proc = st.session_state.get("imagen_procesada", imagen_procesada)

                # Mensaje de éxito
                renderizar_exito()

                espaciador()

                # --- Tarjetas de resultados ---
                emociones_detalle = resultado["emociones_detalle"]
                emocion_max = max(emociones_detalle.values()) if emociones_detalle else 0

                # Tarjeta de Emoción
                renderizar_tarjeta_resultado(
                    tipo="emotion",
                    icono=resultado["emocion_emoji"],
                    etiqueta="Emoci&oacute;n Detectada",
                    valor=resultado["emocion_dominante"],
                    detalle="Emoci&oacute;n dominante seg&uacute;n modelo CNN",
                    confianza=emocion_max,
                )

                espaciador()

                # Tarjeta de Género
                renderizar_tarjeta_resultado(
                    tipo="gender",
                    icono="&#128104;" if resultado["genero_clave"] == "Man" else "&#128105;",
                    etiqueta="G&eacute;nero Detectado",
                    valor=resultado["genero"],
                    detalle="Clasificaci&oacute;n binaria con CNN",
                    confianza=resultado["genero_confianza"],
                )

                espaciador()

                # Tarjeta de Edad
                renderizar_tarjeta_resultado(
                    tipo="age",
                    icono="&#127874;",
                    etiqueta="Edad Estimada",
                    valor=f"{resultado['edad']} a&ntilde;os",
                    detalle="Estimaci&oacute;n por modelo de regresi&oacute;n",
                    confianza=None,
                )

        # ============================================================
        # SECCIÓN INFERIOR: DETALLES Y DESCARGA
        # ============================================================
        if "resultado" in st.session_state and st.session_state["resultado"]["exito"]:
            resultado = st.session_state["resultado"]
            imagen_proc = st.session_state.get("imagen_procesada", imagen_procesada)

            espaciador()

            col_detalle, col_anotada = st.columns([1, 1], gap="large")

            with col_detalle:
                # Distribución de emociones
                # NOTA: renderizar_detalle_emociones genera HTML completo
                # auto-contenido, sin dividir tags entre llamadas.
                renderizar_detalle_emociones(
                    resultado["emociones_detalle"],
                    resultado.get("emocion_clave", "neutral"),
                )

                # --- Métricas rápidas con st.metric ---
                espaciador()
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        label="Emocion",
                        value=resultado["emocion_dominante"].split(" ")[0],
                    )
                with m2:
                    st.metric(
                        label="Genero",
                        value=resultado["genero"].split(" ")[0],
                    )
                with m3:
                    st.metric(
                        label="Edad",
                        value=f"{resultado['edad']}",
                    )

            with col_anotada:
                # Imagen con anotaciones visuales
                st.markdown(
                    '<div class="glass-card">'
                    '<p class="detail-title"><span>&#127919;</span> Rostro Detectado</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                # Dibujar anotaciones sobre la imagen
                imagen_anotada = dibujar_anotaciones(imagen_proc, resultado)
                st.image(
                    imagen_anotada,
                    caption="Imagen con deteccion de rostro",
                    use_container_width=True,
                )

                # --- Botón de descarga ---
                imagen_bytes = imagen_a_bytes(imagen_anotada)
                st.download_button(
                    label="Descargar imagen analizada",
                    data=imagen_bytes,
                    file_name="faceinsight_resultado.png",
                    mime="image/png",
                    use_container_width=True,
                )

    else:
        # --- Estado inicial: mensaje de bienvenida ---
        espaciador()
        st.markdown(
            '<div class="glass-card" style="text-align: center; padding: 2.5rem;">'
            '<span style="font-size: 3rem; display: block; margin-bottom: 1rem;"><span>&#9757;</span></span>'
            '<p style="color: #94A3B8; font-size: 1.1rem; font-weight: 400; margin: 0;">'
            'Sube una imagen para comenzar el an&aacute;lisis facial'
            '</p>'
            '<p style="color: #64748B; font-size: 0.85rem; margin-top: 0.5rem;">'
            'La IA analizar&aacute; el rostro y detectar&aacute; emoci&oacute;n, g&eacute;nero y edad'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Pie de página ---
    renderizar_pie()


# ============================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ============================================================
if __name__ == "__main__":
    main()
