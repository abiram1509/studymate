import hashlib
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from core import Store, Index, extract_pdf, answer

load_dotenv()
st.set_page_config(page_title='StudyMate',page_icon='📚',layout='wide')
ROOT=Path(__file__).parent
DB=Path(os.getenv('STUDYMATE_DATA_DIR',str(ROOT/'data')))/'studymate.db'
store=Store(DB)

@st.cache_resource
def model_loader():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(max_entries=8)
def get_index(db_path,ids,semantic):
    embedder=model_loader() if semantic else None
    return Index(Store(db_path),ids,embedder)

@st.cache_resource
def provider(key):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key,http_options=types.HttpOptions(timeout=20000,retry_options=types.HttpRetryOptions(attempts=1)))

st.title('📚 StudyMate')
st.caption('Ask questions across your PDFs, inspect page sources, and return to saved conversations.')
with st.sidebar:
    st.header('Your library')
    st.caption('Local workspace: documents and history are shared by users of this installation. Do not expose it publicly without authentication.')
    uploads=st.file_uploader('Add PDFs',type='pdf',accept_multiple_files=True)
    if st.button('Process PDFs',disabled=not uploads):
        for upload in uploads:
            if upload.size>25*1024*1024:
                st.error(f'{upload.name}: exceeds the 25 MB limit.');continue
            try:
                with st.spinner(f'Processing {upload.name}...'):
                    _,chunks,warnings,hit=store.put_document(upload.name,upload.getvalue(),extract_pdf)
                st.success(f'{upload.name}: {len(chunks)} passages'+(' (cached)' if hit else ''))
                for warning in warnings:st.warning(warning)
            except ValueError as e:st.error(str(e))
    docs=dict(store.documents())
    ids=st.multiselect('Search documents',list(docs),default=list(docs),format_func=lambda x:docs[x])
    semantic=st.checkbox('Semantic search',value=os.getenv('ENABLE_SEMANTIC','0')=='1',help='Requires optional sentence-transformers installation and an initial model download. Otherwise keyword retrieval works offline.')
    threshold=st.slider('Minimum relevance',0.0,1.0,.30,.05)
    mode=st.radio('Mode',['PDF Q&A','Flashcards','Practice quiz','General chat'])
    if docs:
        with st.expander('Remove a document'):
            remove=st.selectbox('Document',list(docs),format_func=lambda x:docs[x])
            st.caption('Also clears saved conversations and answer caches that could contain document text.')
            if st.button('Delete document and clear history'):
                store.delete_document(remove);get_index.clear();st.rerun()

workspace=hashlib.sha256((mode+'|'+','.join(sorted(ids))).encode()).hexdigest()
key=os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
model=os.getenv('GEMINI_MODEL','gemini-2.5-flash')
generate=None
if key:
    try:
        client=provider(key)
        def generate(prompt):
            return client.models.generate_content(model=model,contents=prompt).text or ''
    except Exception:st.warning('AI connection could not initialize. PDF search is available offline.')
else:st.info('Offline mode: add GOOGLE_API_KEY to .env for AI answers. PDF search works without a key.')

def show_sources(sources,prefix):
    for n,source in enumerate(sources,1):
        with st.expander(f'[{n}] {source["name"]} · Page {source["page"]}'):
            st.write(source['text'])
            if source['document'] in docs:
                name,_,data=store.document(source['document'])
                st.download_button('Download source PDF',data,file_name=name,mime='application/pdf',key=f'{prefix}-{n}')
                if st.button('Preview cited page',key=f'preview-{prefix}-{n}'):
                    import pymupdf as fitz
                    with fitz.open(stream=data,filetype='pdf') as pdf:
                        pix=pdf[source['page']-1].get_pixmap(matrix=fitz.Matrix(1,1))
                        st.image(pix.tobytes('png'),caption=f'{name} - page {source["page"]}')

history=store.history(workspace)
for i,(question,response,sources) in enumerate(history):
    with st.chat_message('user'):st.write(question)
    with st.chat_message('assistant'):
        st.write(response);show_sources(sources,f'history-{i}')
if history:
    transcript='\n\n'.join(f'Q: {q}\nA: {a}' for q,a,_ in history)
    st.download_button('Export conversation',transcript,file_name='studymate-chat.txt')
    if st.button('Clear this conversation'):store.clear_history(workspace);st.rerun()
st.caption('For Flashcards or Practice quiz, enter a topic from your PDFs. Generated study material includes source citations.')
question=st.chat_input('Ask a question or enter a study topic',max_chars=2000)
if question:
    if mode!='General chat' and not ids:
        st.warning('Upload and select at least one PDF first.')
    else:
        with st.spinner('Finding an answer...'):
            sources=[]
            if mode!='General chat':
                try: index=get_index(str(DB),tuple(sorted(ids)),semantic)
                except Exception:
                    st.warning('Semantic model unavailable; using offline keyword search.')
                    index=get_index(str(DB),tuple(sorted(ids)),False)
                sources=index.search(question,threshold=threshold)
            response,status=answer(store,workspace,question,sources,generate,model,mode)
            store.save_message(workspace,question,response,sources)
        st.rerun()
