import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Antes de rodar, certifique-se de ter suas chaves de API do OpenAI configuradas nas variáveis de ambiente:
# export OPENAI_API_KEY='sua_chave_aqui'
# Ou defina com: os.environ["OPENAI_API_KEY"] = "sua_chave_aqui"

########################################
# Configurações
########################################
BASE_URL = "https://www.infomoney.com.br/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
# Parâmetros do LangChain
CHROMA_DB_DIR = "chroma_db"
EMBEDDINGS_MODEL = OpenAIEmbeddings()

########################################
# Funções de Web Scraping
########################################
def get_all_links(url):
    """Extrai todos os links internos do site."""
    page = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(page.content, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        # Ajuste a lógica para capturar apenas links internos
        if href.startswith("/") and not href.startswith("//"):
            full_link = BASE_URL + href
            links.add(full_link)
        elif href.startswith(BASE_URL):
            links.add(href)

    return links

def get_text_content(url):
    """Extrai o texto principal da página."""
    page = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(page.content, "html.parser")
    # Ajuste caso queira extrair algo mais específico, por exemplo, texto de p, div...
    text = " ".join([p.get_text(strip=True) for p in soup.find_all('p')])
    return text.strip()

print(get_all_links(BASE_URL))

########################################
# Coletando Páginas
########################################
visited = set()
to_visit = {BASE_URL}
all_pages_content = []

# Exemplo simples: BFS no domínio (tome cuidado com sites muito grandes)
while to_visit:
    current_url = to_visit.pop()
    if current_url not in visited:
        visited.add(current_url)
        try:
            text = get_text_content(current_url)
            if text:
                all_pages_content.append((current_url, text))
                print("Adding page:", current_url)
            # Descobrir novos links
            new_links = get_all_links(current_url)
            for link in new_links:
                if link not in visited:
                    to_visit.add(link)
        except Exception as e:
            print(f"Erro ao processar {current_url}: {e}")
            
print(f"Coletadas {len(all_pages_content)} páginas.")

# ########################################
# # Pré-Processamento e Indexação no Chroma
# ########################################
# # Dividir os documentos em chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# documents = []
# for url, content in all_pages_content:
#     for chunk in text_splitter.split_text(content):
#         # Metadados podem ser úteis para referência
#         documents.append({"page_content": chunk, "metadata": {"source": url}})

# # Cria o index local
# os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# vectorstore = Chroma.from_documents(documents, EMBEDDINGS_MODEL, persist_directory=CHROMA_DB_DIR)
# vectorstore.persist()

# ########################################
# # Cria a chain de RAG (Retriever Augmented Generation)
# ########################################
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
# qa_chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(temperature=0),
#     chain_type="stuff",
#     retriever=retriever
# )

# ########################################
# # Uso
# ########################################
# # Agora você pode fazer perguntas ao modelo, que utilizará as páginas indexadas
# query = "Digite aqui a sua pergunta sobre o conteúdo do site."
# resposta = qa_chain.run(query)
# print("Resposta:", resposta)