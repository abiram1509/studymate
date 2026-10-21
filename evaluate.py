import json
import tempfile
from pathlib import Path
from core import Store, Index, chunk_pages

def evaluate():
    cases=[('What process converts sunlight into energy?',1),('What attracts objects with mass?',2),('Where is DNA stored?',3)]
    with tempfile.TemporaryDirectory() as tmp:
        store=Store(Path(tmp)/'eval.db')
        pages=[(1,'Photosynthesis converts sunlight into chemical energy in plants.'),(2,'Gravity attracts objects with mass.'),(3,'DNA is stored in the cell nucleus.')]
        key,*_=store.put_document('synthetic-science.pdf',b'evaluation-v1',lambda _: (chunk_pages(pages),[]))
        index=Index(store,[key]);results=[]
        for question,page in cases:
            hits=index.search(question)
            results.append({'question':question,'expected_page':page,'retrieved_page':hits[0]['page'] if hits else None,'pass':bool(hits and hits[0]['page']==page)})
        no_match=not index.search('volcano magma eruption')
    return {'scope':'Small synthetic keyword retrieval smoke evaluation; not real-world answer accuracy.','cases':results,'irrelevant_query_rejected':no_match,'passed':sum(r['pass'] for r in results)+int(no_match),'total':4}
if __name__=='__main__':print(json.dumps(evaluate(),indent=2))
