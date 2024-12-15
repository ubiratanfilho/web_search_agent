import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
import dotenv

dotenv.load_dotenv()

# Inicializar o modelo de linguagem da OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Configurar a ferramenta de busca com SerpAPI
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=lambda query: search.run(f"site:infomoney.com.br {query}"),
    description="Realiza buscas na internet."
)

# Inicializar o agente com a ferramenta de busca
tools = [search_tool]
agent = initialize_agent(tools, llm, agent="openai-functions", verbose=True)

# Função para buscar informações sobre uma empresa
def buscar(query):
    resposta = agent.run(query)
    return resposta

# Exemplo de uso
query = "Quais últimas notícias da ISA ENERGIA BRASIL?"
informacoes = buscar(query)

print(informacoes)