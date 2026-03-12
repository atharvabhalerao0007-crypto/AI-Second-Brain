from pathlib import Path
from typing import List
import PyPDF2


class DocumentLoader:
    def __init__(self, doc_dir: str):
        self.doc_dir = Path(doc_dir)

    def load_documents(self) -> List[str]:
        texts = []

        for file in self.doc_dir.glob("*.pdf"):
            texts.extend(self._read_pdf(file))   # extend instead of append

        return texts

    def _read_pdf(self, file_path: Path) -> List[str]:

        pages_text = []

        with open(file_path, "rb") as f:

            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:
                    page_content = f"Page {page_num}: {text}"

                    pages_text.append(page_content)

        return pages_text