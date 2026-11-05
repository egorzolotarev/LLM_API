from openai import AsyncOpenAI
import httpx
from dotenv import load_dotenv
import os
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import numpy as np
import asyncio

_embedding_model = None

def _get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedding_model

class LLMAPI():

    def __init__(self):
        load_dotenv()
        self._api_key = os.getenv("API_KEY")
        self._clients = {}

        self.data=None
        self.rag=None
        self._model=_get_model()
        self._doc_embeddings=None
        self._chunk_data=None
    
    def _get_client(self):
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            self._clients = {l: c for l, c in self._clients.items() if not l.is_closed()}
            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url="https://api.intelligence.io.solutions/api/v1/",
            )
            self._clients[loop] = client
        return client

    async def _get_rag_from_url(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return

        self.data = await asyncio.to_thread(self.__extract_text, response.text)
        await asyncio.to_thread(self.__generate_RAG)
    
    def get_rag_from_url(self, url):
        coro = self._get_rag_from_url(url)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro
    
    def __extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(strip=True)
    
    def clear_rag(self):
        self.data=None
        self.rag=None
    
    def __generate_RAG(self):
        chunker = RecursiveChunker(chunk_size=1024)
        chunks = chunker.chunk(self.data)
        self._chunk_data = [c.text for c in chunks]

        self._doc_embeddings = self._model.encode(self._chunk_data, normalize_embeddings=True)

    def __generate_message_RAG(self, message):
        m = self._model.encode([message], normalize_embeddings=True)[0]    
        scores = self._doc_embeddings @ m
        best = np.argsort(scores)[::-1][:5]

        self.rag = "Ответь на вопрос, опираясь только на контекст ниже. Если ответа в контексте нет, так и скажи. Контекст: "
        for i in best:
            self.rag += self._chunk_data[i] + " "

    async def _send_message(self, message):
        if self.data == None:
            self.rag=""
        else:
            await asyncio.to_thread(self.__generate_message_RAG, message)
        
        client = self._get_client()
        responce = await client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it",
            messages=[
                {"role": "system", "content": self.rag},
                {"role": "user", "content": message}
            ]
        )

        return [responce.choices[0].message.content, self.rag]
    
    def send_message(self, message):
        coro = self._send_message(message)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro