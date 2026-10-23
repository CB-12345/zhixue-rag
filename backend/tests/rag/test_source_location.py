"""页码从解析、证据上下文到公开来源的契约回归。"""
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.graph.schemas.graph import ChunkHit
from app.rag.agentic_rag.evidence import prepare_evidence_context
from app.rag.agentic_rag.local_retriever import LocalRetriever
from app.rag.agentic_rag.schemas import Evidence
from app.rag.agentic_rag.sources import public_sources
from app.utils import file_handler


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("password", [None, "fixture-password"])
async def test_real_pdf_fallback_uses_physical_one_based_pages(tmp_path, monkeypatch, asynchronous, password):
    import fitz

    path = tmp_path / "two-pages.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "First page: introduction.")
        pdf.new_page().insert_text((72, 72), "Second page: LOC-5813 owner is Lin Qiao.")
        options = {"encryption": fitz.PDF_ENCRYPT_AES_256, "user_pw": password,
                   "owner_pw": "fixture-owner"} if password else {}
        pdf.save(path, **options)

    def unavailable(*args, **kwargs):
        raise RuntimeError("force fallback to PyPDFLoader")

    monkeypatch.setattr(file_handler, "UnstructuredPDFLoader", unavailable)
    docs = (await file_handler.pdf_loader(str(path), password) if asynchronous
            else file_handler.pdf_loader_sync(str(path), password))
    assert [doc.metadata["page"] for doc in docs] == [1, 2]
    chunks = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0).split_documents(docs)
    fact = next(chunk for chunk in chunks if "LOC-5813" in chunk.page_content)
    assert fact.metadata["page"] == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_pdf_without_page_metadata_does_not_invent_a_page(monkeypatch, asynchronous):
    class WholeDocumentLoader:
        def __init__(self, *args):
            pass

        def load(self):
            return [Document(page_content="Whole PDF text", metadata={"source": "a.pdf"})]

    monkeypatch.setattr(file_handler, "UnstructuredPDFLoader", WholeDocumentLoader)
    docs = (await file_handler.pdf_loader("a.pdf") if asynchronous
            else file_handler.pdf_loader_sync("a.pdf"))
    assert "page" not in docs[0].metadata


def test_canonical_chunk_location_reaches_context_and_source_snapshot():
    hit = ChunkHit(id="doc:md5:7", kind="doc", source_id="md5", source_name="guide.pdf",
                   chunk_index=7, text="LOC-5813 owner is Lin Qiao.", page=2,
                   metadata={"source_id": "wrong", "page": 99, "chunk_index": 999})
    evidence = LocalRetriever._chunk_to_evidence(hit)
    prepared = prepare_evidence_context([evidence], max_chars=100)
    sources = public_sources(prepared.evidences)
    assert "（第 2 页）" in prepared.text
    assert len(prepared.text) <= 100
    assert sources[0]["page"] == 2
    assert sources[0]["source_id"] == "md5"
    assert sources[0]["chunk_index"] == 7
    assert sources[0]["id"] == "doc:md5:7"
    assert prepared.text.endswith(sources[0]["content"])


@pytest.mark.parametrize("page", [None, 0, -1, "2", True, 1.5])
def test_unknown_or_invalid_pages_are_not_displayed(page):
    evidence = Evidence(id="c", source="knowledge_base", title="guide.pdf", content="fact",
                        metadata={"page": page})
    assert public_sources([evidence])[0]["page"] is None
    assert " 页）" not in prepare_evidence_context([evidence]).text
