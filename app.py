import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import importlib
from io import BytesIO
import os

st.set_page_config(page_title="Visão Computacional - Ermesson Lima", page_icon="🔍")

st.markdown(
    """
    <style>
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

QUESTOES = {
    "Questão 1 - Filtros de Convolução": "questions.question1",
    "Questão 2 - Filtro Passa-Alta": "questions.question2",
    "Questão 3 - Filtro Bilateral": "questions.question3",
    "Questão 4 - Filtro de Sobel": "questions.question4",
    "Questão 5 - Sobel + Gaussian Blur": "questions.question5",
    "Questão 6 - Detector de Borda Canny": "questions.question6",
    "Questão 7 - Pirâmides Gaussianas e de Downsampling": "questions.question7",
    "Questão 8 - Redimensionamento de Imagens": "questions.question8",
}

IMAGE_DIR = "images/"
IMAGES = {
    "Sinal de Trânsito": "sign_1.ppm",
    "Árvore": "tree_1.ppm",
    "Rua": "west_2.ppm",
}


def read_ppm(filepath): 
    with open(filepath, "rb") as file:
        content = file.read()
        lines = content.splitlines()

        header = lines[0]
        if header not in [b"P3", b"P6"]:
            raise ValueError("Apenas imagens P3 e P6 são suportadas!")

        width, height = map(int, lines[1].split())
        max_val = int(lines[2])

        if header == b"P3":
            pixels = list(map(int, b" ".join(lines[3:]).split()))
            image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        else:
            pixel_data = content.split(b"\n", 3)[-1]
            image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                (height, width, 3)
            )

    return image

def save_ppm(image): 
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)  
    
    height, width, _ = image.shape
    buffer = BytesIO()

    buffer.write(b"P3\n")
    buffer.write(f"{width} {height}\n255\n".encode())

    for row in image:
        for pixel in row:
            buffer.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ".encode())
        buffer.write(b"\n")

    buffer.seek(0)
    return buffer


st.title("Processamento de Imagens com Filtros e Transformações")
st.subheader("Universidade Federal de Alagoas - Instituto de Computação")
st.write(
    """
 **Aluno:** Ermesson Lima  
 **Disciplina:** Visão Computacional  
 **Professor:** Thales Vieira  
 **1ª Lista de Exercícios - 2025**  
"""
)

st.markdown("### Selecione ou faça upload de uma imagem PPM")

col1, col2 = st.columns([1, 2])
with col1:
    modo = st.radio(
        "Escolha a origem da imagem:", ("Usar imagem pronta", "Fazer upload")
    )

if modo == "Usar imagem pronta":
    imagem_escolhida = st.selectbox("Escolha uma imagem:", list(IMAGES.keys()))
    image_path = os.path.join(IMAGE_DIR, IMAGES[imagem_escolhida])
    image = read_ppm(image_path)
else:
    uploaded_file = st.file_uploader("Faça upload de uma imagem PPM", type=["ppm"])
    if uploaded_file is not None:
        image = read_ppm(uploaded_file)


questao_escolhida = st.selectbox(
    "Escolha a questão para aplicar", list(QUESTOES.keys())
)

if "image" in locals():
    modulo_questao = importlib.import_module(QUESTOES[questao_escolhida])

    if hasattr(modulo_questao, "run"):
        resultado = modulo_questao.run(image)

        fig, axes = plt.subplots(1, len(resultado) + 1, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Imagem Original")
        axes[0].axis("off")

        for i, img in enumerate(resultado):
            axes[i + 1].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[i + 1].set_title(f"Resultado {i+1}")
            axes[i + 1].axis("off")

            ppm_buffer = save_ppm(img)
            st.download_button(
                label=f"Baixar Resultado {i+1}",
                data=ppm_buffer,
                file_name=f"resultado_{i+1}.ppm",
                mime="application/octet-stream",
            )

        st.pyplot(fig)
    else:
        st.error(f"A questão '{questao_escolhida}' não possui a função 'run()'.")
