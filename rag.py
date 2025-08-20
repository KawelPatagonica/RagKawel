
import os
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer

# --- OPCIONAL: forzar cache en SSD ---
os.environ['HF_HOME'] = r"D:\Kawel Patagonica\huggingface_cache"

# 1. Leer PDF
# Modifica esta función para que reciba la ruta de una carpeta
def leer_pdfs_en_carpeta(ruta_carpeta):
    texto_total = ""  # Variable para acumular el texto de todos los PDFs
    
    # 1.1. Itera sobre los archivos de la carpeta
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith(".pdf"):  # Asegúrate de que el archivo sea un PDF
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            print(f"Leyendo: {nombre_archivo}")
            
            # 1.2. Procesa cada PDF individualmente
            lector = PdfReader(ruta_completa)
            for pagina in lector.pages:
                pagina_texto = pagina.extract_text()
                if pagina_texto:
                    texto_total += pagina_texto + " "
                    
    return texto_total

# 2. Crear embeddings y FAISS index
def crear_index(texto, model):
    oraciones = texto.split(". ")
    embeddings = model.encode(oraciones, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return oraciones, index

# 3. Buscar contexto más relevante
def buscar_contexto(pregunta, oraciones, index, model_emb, k=3):
    q_emb = model_emb.encode([pregunta], convert_to_numpy=True)
    _, I = index.search(q_emb, k=k)
    contexto = " ".join([oraciones[i] for i in I[0]])
    return contexto

# --- MAIN ---
if __name__ == "__main__":
    # La ruta ahora apunta a la carpeta, no a un solo archivo
    ruta_carpeta_pdfs = r"D:\Kawel Patagonica\RAG_Kawel\pdfs"
    
    # Llama a la nueva función que lee todos los PDFs
    texto = leer_pdfs_en_carpeta(ruta_carpeta_pdfs)

    print("Creando embeddings (CPU)...")
    modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")
    oraciones, index = crear_index(texto, modelo_emb)
    print("Listo. Puedes hacer preguntas.")

    while True:
        pregunta = input("\nTu pregunta (o 'salir'): ")
        if pregunta.lower() == "salir":
            break
        respuesta = buscar_contexto(pregunta, oraciones, index, modelo_emb)
        print(">> Contexto relevante:\n", respuesta)

