from rag import Chunk, split_chunks

def test_chunking_preserves_metadata():
    chunks = split_chunks([Chunk('a' * 1200, 'a.txt', None)], size=500, overlap=100)
    assert len(chunks) > 1
    assert all(c.filename == 'a.txt' for c in chunks)
