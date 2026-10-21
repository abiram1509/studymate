import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from core import Store,chunk_pages

class UITests(unittest.TestCase):
    def test_offline_question_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'studymate.db')
            store.put_document('science.pdf',b'fixture',lambda _: (chunk_pages([(1,'Photosynthesis converts sunlight into chemical energy.')]),[]))
            with patch.dict(os.environ,{'STUDYMATE_DATA_DIR':tmp,'GOOGLE_API_KEY':'','GEMINI_API_KEY':'','ENABLE_SEMANTIC':'0'}):
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
                self.assertEqual(len(app.exception),0)
                app.chat_input[0].set_value('What converts sunlight into chemical energy?').run(timeout=30)
                self.assertEqual(len(app.exception),0)
                self.assertTrue(any('retrieved excerpts' in m.value for m in app.markdown))
                again=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
                self.assertEqual(len(again.exception),0)
                self.assertTrue(any('retrieved excerpts' in m.value for m in again.markdown))
if __name__=='__main__':unittest.main()
