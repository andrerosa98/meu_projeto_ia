import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Imports para Agentes e Ferramentas
from langchain.agents import AgentExecutor, create_react_agent # ReAct é um tipo de agente
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool # Para transformar seu RAG em uma ferramenta
from langchain import hub # Para obter prompts de agente pré-definidos

# Carregar variáveis de ambiente
load_dotenv()

# --- 1. Configurações Iniciais ---
PDF_PATH = "MCASP.pdf"
MODEL_NAME_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index_orcamento"
# Nome do modelo Gemini que funcionou para você (ajuste se necessário)
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest" # ou "gemini-pro" se esse funcionou

# --- 2. Função para criar ou carregar o índice vetorial (semelhante à anterior) ---
def get_vector_store(pdf_path, embeddings_model, index_path):
    if os.path.exists(index_path):
        print(f"Carregando índice FAISS de '{index_path}'...")
        # É importante passar allow_dangerous_deserialization=True se o índice foi criado com uma versão de FAISS
        # que pode ter problemas de segurança com pickles, mas certifique-se que você confia na origem do índice.
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
        print("Índice carregado.")
    else:
        print(f"Criando índice FAISS para '{pdf_path}'...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print(f"Nenhum documento carregado do PDF: {pdf_path}.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Aumentei um pouco o overlap
        texts = text_splitter.split_documents(documents)
        if not texts:
            print("Nenhum texto para indexar após o split.")
            return None
        print(f"Número de chunks de texto criados: {len(texts)}")
        vector_store = FAISS.from_documents(texts, embeddings_model)
        vector_store.save_local(index_path)
        print(f"Índice FAISS salvo em '{index_path}'.")
    return vector_store

# --- 3. Função principal para executar a IA com Agente ---
def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not google_api_key:
        print("Erro: GOOGLE_API_KEY não encontrada.")
        return
    if not tavily_api_key:
        print("Erro: TAVILY_API_KEY não encontrada. A busca na web não funcionará.")
        # Você pode decidir se quer continuar sem busca na web ou parar.
        # Por enquanto, vamos permitir continuar, mas a ferramenta Tavily não será criada.
        # return # Descomente para parar se a chave Tavily for essencial

    print(f"Carregando modelo de embeddings: {MODEL_NAME_EMBEDDINGS}...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDINGS)
    print("Modelo de embeddings carregado.")

    if not os.path.exists(PDF_PATH):
        print(f"Erro: Arquivo PDF '{PDF_PATH}' não encontrado.")
        return

    vector_store = get_vector_store(PDF_PATH, embeddings, INDEX_PATH)
    if not vector_store:
        print("Não foi possível criar ou carregar o vector store. Encerrando.")
        return

    print("Inicializando LLM (Gemini)...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=google_api_key,
                                 temperature=0.2, # Temperatura um pouco mais baixa para agentes pode ser bom
                                 convert_system_message_to_human=True)
    print("LLM inicializado.")

    # --- Criar Ferramentas para o Agente ---
    tools = []

    # Ferramenta 1: Retriever dos seus documentos PDF
    pdf_retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # Recuperar 4 chunks
    retriever_tool = create_retriever_tool(
        pdf_retriever,
        "busca_documento_orcamento", # Nome da ferramenta
        "Busca informações nos documentos carregados sobre orçamento, leis e manuais. Use esta ferramenta para perguntas específicas sobre o conteúdo desses documentos."
    )
    tools.append(retriever_tool)

    # Ferramenta 2: Busca na Web com Tavily (somente se a chave API estiver disponível)
    if tavily_api_key:
        tavily_tool = TavilySearchResults(max_results=3) # Retorna os 3 melhores resultados
        tavily_tool.name = "busca_na_web_geral" # Dando um nome mais descritivo
        tavily_tool.description = "Uma ferramenta de busca na web para encontrar informações gerais, notícias atuais ou tópicos não cobertos pelos documentos de orçamento. Use para perguntas mais amplas ou que necessitem de conhecimento externo."
        tools.append(tavily_tool)
    else:
        print("Chave da API Tavily não encontrada. A ferramenta de busca na web não será adicionada.")


    if not tools:
        print("Nenhuma ferramenta foi configurada para o agente. Encerrando.")
        return

    # --- Criar o Agente ---
    # Obter um prompt pré-definido do LangChain Hub para agentes ReAct com chat.
    # Este prompt é projetado para funcionar bem com modelos de chat e o formato ReAct.
    # O prompt "hwchase17/react-chat" é uma boa escolha geral.
    # Você pode explorar outros no hub: https://smith.langchain.com/hub
    try:
        prompt = hub.pull("hwchase17/react") # Um prompt popular para agentes de chat, sem o -chat
    except Exception as e:
        print(f"Erro ao puxar o prompt do LangChain Hub: {e}")
        print("Verifique sua conexão com a internet ou se o prompt 'hwchase17/react-chat' ainda está disponível.")
        print("Usando um prompt padrão básico como fallback (pode não ser ideal).")
        # Fallback para um prompt mais simples se o hub falhar
        from langchain.prompts import PromptTemplate
        # Este é um exemplo muito básico, o do hub é melhor
        template = """Responda à seguinte pergunta da melhor forma possível. Você tem acesso às seguintes ferramentas:
        {tools}
        Use o seguinte formato:
        Pergunta: a pergunta que você deve responder
        Pensamento: você deve sempre pensar sobre o que fazer
        Ação: a ação a ser tomada, deve ser uma de [{tool_names}]
        Entrada da Ação: a entrada para a ação
        Observação: o resultado da ação
        ... (este Pensamento/Ação/Entrada da Ação/Observação pode se repetir N vezes)
        Pensamento: Eu agora sei a resposta final
        Resposta Final: a resposta final para a pergunta 
        Suas respostas devem ser todas em português do Brasil. Assuma que o usuário é brasileiro. As respostas devem ser sérias e baseadas em fatos.    
        Quando não souber, pesquise na internet, mas se atenha aos documentos fornecidos.

        Comece!

        Pergunta: {input}
        {agent_scratchpad}"""
        prompt = PromptTemplate.from_template(template)


    agent = create_react_agent(llm, tools, prompt)

    # Criar o Executor do Agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # verbose=True mostra os "pensamentos" do agente, ótimo para depuração
        handle_parsing_errors=True, # Tenta lidar com erros de formatação da saída do LLM
        max_iterations=5 # Evita loops infinitos do agente
        )

    print("\n--- Agente de IA de Orçamento Pronto ---")
    print("Pergunte sobre seus documentos ou peça buscas na web (ou digite 'sair' para terminar)")

    while True:
        user_input = input("\nSua pergunta: ")
        if user_input.lower() == 'sair':
            break
        if not user_input.strip():
            continue

        print("Agente processando sua pergunta...")
        try:
            # Para o agente, o input é geralmente uma string simples.
            response = agent_executor.invoke({"input": user_input})
            print("\nResposta Final do Agente:")
            print(response["output"])

        except Exception as e:
            print(f"Ocorreu um erro ao executar o agente: {e}")
            # Imprimir o traceback pode ser útil para depuração mais profunda
            # import traceback
            # traceback.print_exc()

    print("Encerrando a IA de Orçamento.")

if __name__ == "__main__":
    main()