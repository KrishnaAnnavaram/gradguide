"""Problem 7: re-indexing duplicated or kept stale chunks; ingestion re-embedded everything."""
import shutil

from conftest import SAMPLE_DOCS, CountingEmbedder

from gradguide.index.store import Index, sync_index


def _copy_docs(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DOCS, docs)
    return docs


def test_second_run_loads_cache_without_embedding(tmp_path, embedder):
    index, report = sync_index(SAMPLE_DOCS, tmp_path / "idx", embedder)
    assert report.added == len(index.chunks) and embedder.batches == [len(index.chunks)]
    again, report = sync_index(SAMPLE_DOCS, tmp_path / "idx", embedder)
    assert report.loaded_from_cache and embedder.batches == [len(index.chunks)]
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in index.chunks]


def test_edit_reembeds_only_changed_chunks_and_drops_stale_text(tmp_path, embedder):
    docs = _copy_docs(tmp_path)
    index, _ = sync_index(docs, tmp_path / "idx", embedder)
    path = docs / "graduation.md"
    path.write_text(path.read_text(encoding="utf-8").replace("$75", "$90"), encoding="utf-8")
    updated, report = sync_index(docs, tmp_path / "idx", embedder)
    assert report.added == 1 and report.removed == 1 and report.kept == len(index.chunks) - 1
    assert embedder.batches[-1] == 1
    texts = " ".join(c.text for c in updated.chunks)
    assert "$90" in texts and "$75" not in texts
    assert len({c.chunk_id for c in updated.chunks}) == len(updated.chunks)  # no duplicates


def test_deleted_document_is_removed(tmp_path, embedder):
    docs = _copy_docs(tmp_path)
    sync_index(docs, tmp_path / "idx", embedder)
    (docs / "assistantships.html").unlink()
    index, report = sync_index(docs, tmp_path / "idx", embedder)
    assert report.removed > 0 and report.added == 0
    assert all(c.source != "assistantships.html" for c in index.chunks)


def test_changing_embedder_reembeds_everything(tmp_path):
    first = CountingEmbedder()
    sync_index(SAMPLE_DOCS, tmp_path / "idx", first)
    other = CountingEmbedder()
    other.name = "hashing-other"
    index, report = sync_index(SAMPLE_DOCS, tmp_path / "idx", other)
    assert report.added == len(index.chunks) and report.kept == 0


def test_saved_index_roundtrip_and_no_vector_json_dump(tmp_path, embedder):
    index, _ = sync_index(SAMPLE_DOCS, tmp_path / "idx", embedder)
    files = sorted(p.name for p in (tmp_path / "idx").iterdir())
    assert files == ["bm25.json", "chunks.jsonl", "manifest.json", "vectors.npy"]
    loaded = Index.load(tmp_path / "idx")
    assert loaded.chunks == index.chunks and loaded.vectors.shape == index.vectors.shape
    assert loaded.manifest["embedder"] == embedder.name and "graduation.md" in loaded.manifest["sources"]
