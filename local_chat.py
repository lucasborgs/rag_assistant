# %% [markdown]
# # 🧠 Local Chat 

# %%
import streamlit as st
import tempfile
import os
import time
import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, NotRequired
from contextlib import contextmanager
from datetime import datetime
import logging

# LangChain & LangGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. Configuração e Constantes ---
class AppConfig:
    def __init__(self):
        self.CACHE_DIR = Path("./vectorstore_cache")
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1.0
        self.MAX_FILE_SIZE_MB = 50
        self.ALLOWED_EXTENSIONS = frozenset([".pdf"])
        self.CHAT_HISTORY_LIMIT = 50
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)

config = AppConfig()

# --- 2. Classes de Estado ---
class GraphState(TypedDict):
    question: str
    generation: NotRequired[str]
    documents: NotRequired[List[Document]]
    error: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]

# Estado da aplicação Streamlit
APP_STATE_DEFAULTS = {
    'vectorstore': None,
    'llm': None,
    'embeddings': None,
    'messages': [],
    'uploaded_files_hash': None,
    'config': {}
}

# --- 3. Utilitários ---
class SecurityValidator:
    @staticmethod
    def validate_file(file_obj) -> tuple[bool, str]:
        try:
            filename = file_obj.name.lower()
            if not filename.endswith('.pdf'):
                return False, "Apenas arquivos PDF são permitidos"
            return True, ""
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        return text.strip()

class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except: self.metadata = {}
        else: self.metadata = {}
    
    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except: pass
    
    def get_cache_path(self, dataset_hash: str) -> Path:
        return self.cache_dir / f"{dataset_hash}.pkl"
    
    def load(self, dataset_hash: str) -> Optional[FAISS]:
        cache_path = self.get_cache_path(dataset_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except: return None
        return None
    
    def save(self, vectorstore: FAISS, dataset_hash: str, metadata: Dict = None):
        cache_path = self.get_cache_path(dataset_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(vectorstore, f)
            self.metadata[dataset_hash] = {
                'created_at': datetime.now().isoformat(),
                'file_count': metadata.get('file_count', 0) if metadata else 0
            }
            self._save_metadata()
        except: pass

# --- 4. Inicialização ---
def init_session_state():
    defaults = {
        'app_state': APP_STATE_DEFAULTS.copy(),
        'cache_manager': CacheManager(config.CACHE_DIR),
        'security_validator': SecurityValidator(),
        'rag_graph': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def ensure_llm_initialized():
    state = st.session_state.app_state
    conf = state.get('config', {})
    if not conf: return

    if not state.get('llm'):
        try:
            state['llm'] = Ollama(
                model=conf.get('llm_model', 'llama3.2'),
                base_url=conf.get('ollama_url', 'http://localhost:11434'),
                temperature=0.2, 
                num_predict=2048
            )
        except Exception as e: logger.error(f"Erro LLM: {e}")

    if not state.get('embeddings'):
        try:
            state['embeddings'] = OllamaEmbeddings(
                model=conf.get('embed_model', 'nomic-embed-text'),
                base_url=conf.get('ollama_url', 'http://localhost:11434'),
                show_progress=False
            )
        except Exception as e: logger.error(f"Erro Embeddings: {e}")

# --- 5. Processamento ---
@contextmanager
def temporary_pdf_file(uploaded_file):
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_file = tmp.name
        yield tmp_file
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try: os.unlink(tmp_file)
            except: pass

def get_dataset_hash(uploaded_files: List, embed_model: str, chunk_size: int) -> str:
    sorted_files = sorted(uploaded_files, key=lambda x: x.name)
    combined_hash = hashlib.sha256()
    combined_hash.update(f"{embed_model}_{chunk_size}".encode())
    for f in sorted_files:
        combined_hash.update(f.name.encode())
    return combined_hash.hexdigest()

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_files(self, uploaded_files: List) -> tuple[List[Document], Dict]:
        all_splits = []
        for file in uploaded_files:
            try:
                with temporary_pdf_file(file) as tmp_path:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = file.name
                    splits = self.splitter.split_documents(docs)
                    all_splits.extend(splits)
            except Exception as e:
                logger.error(f"Erro ao processar {file.name}: {e}")
        return all_splits, {'file_count': len(uploaded_files)}

class VectorStoreManager:
    def __init__(self, embeddings_model, ollama_url):
        self.embeddings = OllamaEmbeddings(
            model=embeddings_model, base_url=ollama_url, show_progress=True
        )
    
    def create_vectorstore(self, documents):
        return FAISS.from_documents(documents, self.embeddings)

# --- 6. Nós do Grafo ---
def retrieve(state: GraphState) -> GraphState:
    """Recupera documentos do Vector Store"""
    try:
        if not st.session_state.app_state.get('vectorstore'):
            return {"question": state["question"], "documents": [], "error": "VectorStore missing"}
        
        # Recuperação
        retriever = st.session_state.app_state['vectorstore'].as_retriever(
            search_kwargs={"k": st.session_state.app_state['config'].get('k_retrieval', 4)}
        )
        documents = retriever.invoke(state["question"])
        
        return {"question": state["question"], "documents": documents}
    except Exception as e:
        return {"question": state["question"], "documents": [], "error": str(e)}

def generate(state: GraphState) -> GraphState:
    """Gera a resposta usando os documentos recuperados"""
    documents = state.get("documents", [])
    question = state.get("question", "")
    
    # Se não houver documentos, tenta responder com o que tem ou avisa
    if not documents:
        return {
            "question": question,
            "generation": "Não encontrei informações relevantes nos documentos carregados para responder a esta pergunta.",
            "documents": []
        }

    # Formatação do Contexto
    context_text = "\n\n".join([
        f"FONTE: {doc.metadata.get('source_file', 'Desconhecido')}\nCONTEÚDO: {doc.page_content}" 
        for doc in documents
    ])

    # Prompt para RAG
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente prestativo e preciso.
        Sua tarefa é responder à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto fornecido abaixo.
        
        Diretrizes:
        1. Se a resposta estiver no contexto, responda detalhadamente em Português.
        2. Cite o nome do arquivo fonte sempre que possível.
        3. Se a resposta NÃO estiver no contexto, diga: "Não encontrei essa informação nos documentos."
        4. Não invente informações.
        
        CONTEXTO DOS DOCUMENTOS:
        {context}"""),
        ("human", "{question}")
    ])
    
    llm = st.session_state.app_state.get('llm')
    if not llm:
        return {"question": question, "generation": "Erro: LLM não conectado."}

    # Cadeia de Geração
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context_text, "question": question})
        return {
            "question": question,
            "generation": response,
            "documents": documents,
            "metadata": {"sources": list(set(d.metadata.get('source_file') for d in documents))}
        }
    except Exception as e:
        return {"question": question, "generation": f"Erro na geração: {e}"}

# --- 7. Construção do grafo linear ---
def build_rag_graph() -> StateGraph:
    """
    Constrói um grafo RAG simples e robusto.
    Fluxo: Retrieve -> Generate
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # Fluxo direto
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# --- 8. Interface Streamlit ---
def main():
    init_session_state()
    st.set_page_config(page_title="DocuChat Local", page_icon="📝", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("📝 Local Chat")
        st.info("Sistema para interpretação de documentos.")
        
        # Configurações
        with st.expander("⚙️ Configurações", expanded=True):
            config_settings = {
                'llm_model': st.selectbox("Modelo", ["llama3.2", "mistral", "qwen2.5"], index=0),
                'embed_model': st.selectbox("Embeddings", ["nomic-embed-text", "all-minilm"], index=0),
                'ollama_url': st.text_input("URL Ollama", "http://localhost:11434"),
                'chunk_size': st.number_input("Tamanho do Chunk", 500, 2000, 1000),
                'k_retrieval': st.slider("Docs Recuperados", 2, 8, 4)
            }
            st.session_state.app_state['config'] = config_settings

        if st.button("🗑️ Limpar Conversa"):
            st.session_state.app_state['messages'] = []
            st.rerun()
            
        if st.button("🔄 Resetar Tudo"):
            st.session_state.clear()
            st.rerun()

    # Inicialização tardia do LLM
    ensure_llm_initialized()

    # Área Principal
    st.title("Chat com Documentos")

    # Upload
    if not st.session_state.app_state.get('vectorstore'):
        uploaded_files = st.file_uploader("Carregue seus PDFs aqui", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("🚀 Processar Documentos"):
                with st.spinner("Lendo, fragmentando e indexando..."):
                    # Hash para Cache
                    ds_hash = get_dataset_hash(uploaded_files, config_settings['embed_model'], config_settings['chunk_size'])
                    cached_vs = st.session_state.cache_manager.load(ds_hash)
                    
                    if cached_vs:
                        st.session_state.app_state['vectorstore'] = cached_vs
                        st.session_state.app_state['uploaded_files_hash'] = ds_hash
                        st.success("Carregado do cache!")
                    else:
                        # Processamento Novo
                        proc = DocumentProcessor(chunk_size=config_settings['chunk_size'])
                        splits, meta = proc.process_files(uploaded_files)
                        
                        if splits:
                            mgr = VectorStoreManager(config_settings['embed_model'], config_settings['ollama_url'])
                            vs = mgr.create_vectorstore(splits)
                            
                            st.session_state.cache_manager.save(vs, ds_hash, meta)
                            st.session_state.app_state['vectorstore'] = vs
                            st.session_state.app_state['uploaded_files_hash'] = ds_hash
                            st.success(f"Indexado com sucesso! {len(splits)} fragmentos.")
                        else:
                            st.error("Erro ao ler documentos.")
                    
                    time.sleep(1)
                    st.rerun()
    
    # Interface de Chat
    else:
        st.success("📚 Base de conhecimento ativa.")
        if st.button("Trocar Arquivos"):
            st.session_state.app_state['vectorstore'] = None
            st.rerun()

        # Histórico
        for msg in st.session_state.app_state['messages']:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if 'sources' in msg:
                    with st.expander("Fontes"):
                        for s in msg['sources']: st.write(f"- {s}")

        # Input
        if prompt := st.chat_input("Faça uma pergunta sobre o conteúdo..."):
            st.session_state.app_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando documentos..."):
                    try:
                        # Inicializa grafo se necessário
                        if not st.session_state.app_state.get('rag_graph'):
                            st.session_state.app_state['rag_graph'] = build_rag_graph()
                        
                        # Execução
                        inputs = {"question": prompt}
                        graph = st.session_state.app_state['rag_graph']
                        final_state = graph.invoke(inputs)
                        
                        response = final_state.get("generation", "Erro na geração.")
                        sources = final_state.get("metadata", {}).get("sources", [])
                        
                        st.markdown(response)
                        
                        # Salva no histórico
                        msg_data = {"role": "assistant", "content": response}
                        if sources: 
                            msg_data["sources"] = sources
                            with st.expander("Fontes Utilizadas"):
                                for s in sources: st.write(f"- {s}")
                                
                        st.session_state.app_state['messages'].append(msg_data)
                        
                    except Exception as e:
                        st.error(f"Ocorreu um erro: {e}")
                        logger.error(f"Erro Chat: {e}", exc_info=True)

if __name__ == "__main__":
    main()