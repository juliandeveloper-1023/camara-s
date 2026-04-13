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
#   Imagen -> Deteccion de Rostro -> Alineacion -> Analisis por CNN ->
#   -> Prediccion de Emocion, Genero y Edad
#
#   Cada tarea (emocion, genero, edad) usa un modelo CNN especifico:
#   - Emociones: Modelo entrenado con FER-2013 (35,887 imagenes)
#   - Genero: Modelo binario entrenado con datos demograficos
#   - Edad: Modelo de regresion para estimacion de edad aparente
#
# JUSTIFICACION DEL USO DE DEEPFACE:
#   1. Precision: Modelos de estado del arte con alta precision.
#   2. Facilidad: Una sola linea de codigo para analisis completo.
#   3. Modelos preentrenados: No requiere entrenamiento ni GPU costosa.
#   4. Comunidad activa: Mantenida y actualizada constantemente.
#
# NOTA DE COMPATIBILIDAD CON STREAMLIT CLOUD:
#   Esta version usa UNICAMENTE componentes nativos de Streamlit
#   (st.title, st.write, st.columns, st.metric, st.progress, etc.)
#   El unico uso de unsafe_allow_html=True es para inyectar CSS.
#   Esto elimina POR COMPLETO el error 'removeChild' de React.
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
# CONFIGURACION DE LA PAGINA DE STREAMLIT
# ============================================================
st.set_page_config(
    page_title="FaceInsight AI - Analisis Facial Inteligente",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CARGA DE ESTILOS CSS (unico uso de unsafe_allow_html)
# ============================================================
def cargar_estilos():
    """
    Carga SOLO el CSS. Este es el unico lugar donde se usa
    unsafe_allow_html=True en toda la aplicacion.
    Inyectar <style> NUNCA causa removeChild porque el navegador
    no crea nodos visibles para <style>.
    """
    ruta_css = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(ruta_css):
        with open(ruta_css, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<style>.stApp{background:linear-gradient(135deg,#0F0A1A,#1A1033,#0D1B2A);font-family:'Inter',sans-serif;}</style>",
            unsafe_allow_html=True,
        )


cargar_estilos()


# ============================================================
# FUNCION PRINCIPAL
# ============================================================
def main():
    """
    Funcion principal que orquesta toda la aplicacion.
    USA SOLO componentes nativos de Streamlit para evitar
    errores de removeChild en el DOM de React.
    """

    # --- Encabezado ---
    st.title("🧠 FaceInsight AI")
    st.caption("Analisis facial inteligente · Emocion · Genero · Edad")
    st.caption("⚡ Powered by DeepFace & Deep Learning")

    st.write("")

    # --- Tarjetas informativas ---
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric(label="🎭 Emociones", value="7 clases")
    with col_info2:
        st.metric(label="👤 Genero & Edad", value="Estimacion IA")
    with col_info3:
        st.metric(label="⚡ Deep Learning", value="Preentrenado")

    st.write("")
    st.write("")

    # ============================================================
    # ZONA DE CARGA DE IMAGEN
    # ============================================================
    st.write("📸 **Arrastra o selecciona una imagen para analizar**")
    st.caption("Formatos soportados: JPG, JPEG, PNG")

    archivo_subido = st.file_uploader(
        "Sube tu imagen aqui",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen con un rostro visible para el analisis facial.",
        label_visibility="collapsed",
    )

    # ============================================================
    # PROCESAMIENTO DE LA IMAGEN
    # ============================================================
    if archivo_subido is not None:
        # --- Cargar y preparar la imagen ---
        imagen_original = cargar_imagen(archivo_subido)
        imagen_procesada = redimensionar_imagen(imagen_original)

        st.write("")

        # --- Diseno de dos columnas: imagen + resultados ---
        col_imagen, col_resultados = st.columns([1, 1], gap="large")

        with col_imagen:
            st.write("🖼️ **Imagen Cargada**")
            st.image(
                imagen_procesada,
                caption=archivo_subido.name,
                use_column_width=True,
            )

        with col_resultados:
            # --- Boton para iniciar el analisis ---
            boton_analizar = st.button(
                "🔍 Analizar Rostro",
                use_container_width=True,
                type="primary",
            )

            if boton_analizar:
                # ==============================================
                # Ejecutar el analisis facial con DeepFace
                # ==============================================
                # DeepFace.analyze() internamente:
                # 1. Detecta rostros con el backend (OpenCV)
                # 2. Alinea el rostro detectado
                # 3. Pasa el rostro por modelos CNN:
                #    - CNN de emociones (FER-2013)
                #    - CNN de genero (clasificacion binaria)
                #    - CNN de edad (regresion)
                # 4. Devuelve las predicciones con confianzas
                # ==============================================
                with st.spinner("🔍 Analizando rostro con inteligencia artificial..."):
                    resultado = analizar_rostro(imagen_procesada)
                    time.sleep(0.5)

                if resultado["exito"]:
                    st.session_state["resultado"] = resultado
                    st.session_state["imagen_procesada"] = imagen_procesada
                else:
                    st.error(f"⚠️ {resultado['error']}")

            # --- Mostrar resultados almacenados en session_state ---
            if "resultado" in st.session_state and st.session_state["resultado"]["exito"]:
                resultado = st.session_state["resultado"]
                imagen_proc = st.session_state.get("imagen_procesada", imagen_procesada)

                st.success("✅ Analisis completado con exito")

                st.write("")

                # --- Resultados con metricas nativas ---
                emociones_detalle = resultado["emociones_detalle"]
                emocion_max = max(emociones_detalle.values()) if emociones_detalle else 0

                # Tarjeta de Emocion
                st.write("---")
                st.write(f"**🎭 Emocion Detectada**")
                st.write(f"### {resultado['emocion_emoji']} {resultado['emocion_dominante']}")
                st.progress(min(emocion_max / 100.0, 1.0))
                st.caption(f"Confianza: {emocion_max:.1f}% — Modelo CNN (FER-2013)")

                st.write("")

                # Tarjeta de Genero
                st.write("---")
                icono_genero = "👨" if resultado["genero_clave"] == "Man" else "👩"
                st.write(f"**{icono_genero} Genero Detectado**")
                st.write(f"### {resultado['genero']}")
                st.progress(min(resultado["genero_confianza"] / 100.0, 1.0))
                st.caption(f"Confianza: {resultado['genero_confianza']:.1f}% — Clasificacion binaria CNN")

                st.write("")

                # Tarjeta de Edad
                st.write("---")
                st.write(f"**🎂 Edad Estimada**")
                st.write(f"### {resultado['edad']} anios")
                st.caption("Estimacion por modelo de regresion")

        # ============================================================
        # SECCION INFERIOR: DETALLES Y DESCARGA
        # ============================================================
        if "resultado" in st.session_state and st.session_state["resultado"]["exito"]:
            resultado = st.session_state["resultado"]
            imagen_proc = st.session_state.get("imagen_procesada", imagen_procesada)

            st.write("")
            st.write("---")

            col_detalle, col_anotada = st.columns([1, 1], gap="large")

            with col_detalle:
                # Distribucion de emociones
                st.write("**📊 Distribucion de Emociones**")
                st.write("")

                emociones_detalle = resultado["emociones_detalle"]
                emociones_ordenadas = sorted(
                    emociones_detalle.items(), key=lambda x: x[1], reverse=True
                )
                max_valor_emo = max(emociones_detalle.values()) if emociones_detalle else 0

                for nombre, porcentaje in emociones_ordenadas:
                    col_nombre, col_barra, col_pct = st.columns([2, 4, 1])
                    with col_nombre:
                        if porcentaje == max_valor_emo:
                            st.write(f"**{nombre}**")
                        else:
                            st.write(nombre)
                    with col_barra:
                        st.progress(min(porcentaje / 100.0, 1.0))
                    with col_pct:
                        st.write(f"{porcentaje:.1f}%")

                st.write("")

                # Metricas rapidas
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
                st.write("**🎯 Rostro Detectado**")

                # Dibujar anotaciones sobre la imagen
                imagen_anotada = dibujar_anotaciones(imagen_proc, resultado)
                st.image(
                    imagen_anotada,
                    caption="Imagen con deteccion de rostro",
                    use_column_width=True,
                )

                # Boton de descarga
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
        st.write("")
        st.write("")

        col_empty1, col_central, col_empty2 = st.columns([1, 2, 1])
        with col_central:
            st.write("")
            st.info(
                "👆 **Sube una imagen para comenzar el analisis facial**\n\n"
                "La IA analizara el rostro y detectara emocion, genero y edad."
            )

    # --- Pie de pagina ---
    st.write("")
    st.write("---")
    st.caption(
        "🧠 FaceInsight AI · Desarrollado con Streamlit + DeepFace + OpenCV · "
        "Proyecto academico — Reconocimiento Facial con Machine Learning · 2025"
    )


# ============================================================
# PUNTO DE ENTRADA DE LA APLICACION
# ============================================================
if __name__ == "__main__":
    main()
