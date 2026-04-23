import asyncio
import logging
from app.rag.retriever import Retriever
from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, VMediaReadOnlyStore
from app import config

logging.basicConfig(level=logging.INFO)

async def test_auto_detection():
    voyage = VoyageEmbedder(api_key=config.VOYAGE_API_KEY)
    qdrant_docs = QdrantStore(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection=config.COLLECTION_DOCS
    )
    qdrant_videos = QdrantStore(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection=config.COLLECTION_VIDEOS
    )
    vmedia_store = VMediaReadOnlyStore(
        url=config.QDRANT_VMEDIA_URL,
        api_key=config.QDRANT_VMEDIA_API_KEY,
        collections=config.VMEDIA_COLLECTIONS
    )
    
    retriever = Retriever(voyage, qdrant_docs, qdrant_videos, vmedia_store)
    
    # Test case 1: Production (Sản xuất)
    print("\n--- Testing 'Sản xuất' detection ---")
    hits = retriever.retrieve("Quy định thi công và an toàn lao động tại công trường là gì?")
    
    # Test case 2: Marketing
    print("\n--- Testing 'Marketing' detection ---")
    hits = retriever.retrieve("Các kênh truyền thông và chiến dịch marketing Q2?")

    # Test case 3: IT (Công nghệ thông tin)
    print("\n--- Testing 'Công nghệ thông tin' detection ---")
    hits = retriever.retrieve("Làm sao để cấu hình vpn và sso cho nhân viên mới?")

if __name__ == "__main__":
    asyncio.run(test_auto_detection())
