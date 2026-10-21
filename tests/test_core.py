import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from core import Store, Index, chunk_pages, answer, extract_pdf

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/'test.db')
    def tearDown(self):self.tmp.cleanup()
    def add(self,name='a.pdf',data=b'a',text='Photosynthesis converts sunlight into chemical energy.'):
        return self.store.put_document(name,data,lambda b:(chunk_pages([(2,text)]),[]))[0]
    def test_cache_survives_restart(self):
        self.add()
        other=Store(self.store.path)
        _,chunks,_,hit=other.put_document('a.pdf',b'a',lambda b:self.fail('extraction ran again'))
        self.assertTrue(hit);self.assertEqual(chunks[0]['page'],2)
    def test_full_text_and_pages(self):
        chunks=chunk_pages([(1,'word '*6000),(2,'lastpage marker')])
        self.assertEqual(chunks[-1]['page'],2)
        self.assertTrue(any('marker' in c['text'] for c in chunks))
        self.assertGreater(len(chunks),12)
    def test_multi_document_and_no_match(self):
        a=self.add();b=self.add('b.pdf',b'b','Gravity attracts objects with mass.')
        idx=Index(self.store,[a,b])
        self.assertEqual(idx.search('gravity')[0]['document'],b)
        self.assertEqual(idx.search('volcano'),[])
    def test_atomic_rate_limit(self):
        with ThreadPoolExecutor(max_workers=12) as pool:
            results=list(pool.map(lambda _:self.store.reserve([('global',8)],now=100),range(20)))
        self.assertEqual(results.count(0),8)
        self.assertEqual(self.store.reserve([('global',8)],now=161),0)
    def test_workspace_limit_does_not_consume_global(self):
        self.store.reserve([('global',2),('w',1)],now=100)
        self.assertGreater(self.store.reserve([('global',2),('w',1)],now=100),0)
        self.assertEqual(self.store.reserve([('global',2),('other',1)],now=100),0)
    def test_fallback_and_cache(self):
        sources=Index(self.store,[self.add()]).search('sunlight')
        text,status=answer(self.store,'w','sunlight',sources)
        self.assertEqual(status,'offline');self.assertIn('retrieved excerpts',text)
        text,status=answer(self.store,'w','sunlight',sources,lambda p:'Sunlight becomes chemical energy [1].')
        self.assertEqual(status,'generated')
        _,status=answer(self.store,'w','sunlight',sources,lambda p:self.fail('cache miss'))
        self.assertEqual(status,'cached')
    def test_retry_and_bad_citations(self):
        sources=Index(self.store,[self.add()]).search('sunlight');calls=[]
        class Busy(Exception):code=429
        def fail(prompt):calls.append(prompt);raise Busy()
        _,status=answer(self.store,'w','q',sources,fail,sleep=lambda n:None)
        self.assertEqual(len(calls),2);self.assertEqual(status,'provider-fallback')
        _,status=answer(self.store,'w','different',sources,lambda p:'Wrong [99]')
        self.assertEqual(status,'citation-fallback')
    def test_no_match_does_not_call_api(self):
        _,status=answer(self.store,'w','q',[],lambda p:self.fail('API called'))
        self.assertEqual(status,'no-match')
    def test_history_and_delete(self):
        doc=self.add();self.store.save_message('w','q','a',[])
        self.assertEqual(len(Store(self.store.path).history('w')),1)
        self.store.delete_document(doc)
        self.assertEqual(self.store.history('w'),[]);self.assertEqual(self.store.documents(),[])
    def test_pdf_all_pages(self):
        import pymupdf as fitz
        with fitz.open() as doc:
            for text in ['Photosynthesis uses sunlight.','Gravity attracts mass.']:
                page=doc.new_page();page.insert_text((72,72),text)
            data=doc.tobytes()
        chunks,warnings=extract_pdf(data)
        self.assertEqual([c['page'] for c in chunks],[1,2]);self.assertEqual(warnings,[])
    def test_embedding_cache(self):
        import numpy as np
        class Embedder:
            calls=0
            def encode(self,texts,normalize_embeddings=True):
                self.calls+=1
                return np.asarray([[1.,0.] for _ in texts],dtype='float32')
        model=Embedder();doc=self.add()
        Index(self.store,[doc],model)
        Index(Store(self.store.path),[doc],model)
        self.assertEqual(model.calls,1)
    def test_rate_limited_fallback(self):
        sources=Index(self.store,[self.add()]).search('sunlight')
        for _ in range(5):self.store.reserve([('workspace:w',5)])
        _,status=answer(self.store,'w','q',sources,lambda p:self.fail('limit bypass'))
        self.assertEqual(status,'rate-limited')
    def test_expired_cache(self):
        self.store.cache('key','old',ttl=-1)
        self.assertIsNone(self.store.cached('key'))
    def test_study_modes_grounded(self):
        sources=Index(self.store,[self.add()]).search('sunlight')
        prompts=[]
        def gen(prompt):prompts.append(prompt);return 'Question and answer [1]'
        answer(self.store,'w','sunlight',sources,gen,mode='Flashcards')
        self.assertIn('flashcards',prompts[0])
        _,status=answer(self.store,'w','unknown',[],gen,mode='Practice quiz')
        self.assertEqual(status,'no-match')
    def test_invalid_pdf(self):
        with self.assertRaises(ValueError):extract_pdf(b'broken')

if __name__=='__main__':unittest.main()
