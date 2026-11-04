import openai
import requests
from dotenv import load_dotenv
import os
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import numpy as np

class LLMAPI():

    def __init__(self):
        self.data=None
        self.rag=None
        self._model=None
        self._doc_embeddings=None
        self._chunk_data=None

    def get_rag_from_url(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        self.data = soup.get_text(strip=True)
        self.__generate_RAG()
    
    def clear_rag(self):
        self.data=None
        self.rag=None
    
    def __generate_RAG(self):
        chunker = RecursiveChunker(chunk_size=1024)
        chunks = chunker.chunk(self.data)

        self._chunk_data = [c.text for c in chunks]

        self._model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self._doc_embeddings = self._model.encode(self._chunk_data, normalize_embeddings=True)

    def __generate_message_RAG(self, message):
        m = self._model.encode([message], normalize_embeddings=True)[0]    
        scores = self._doc_embeddings @ m
        best = np.argsort(scores)[::-1][:5]

        self.rag = "Ответь на вопрос, опираясь только на контекст ниже. Если ответа в контексте нет, так и скажи. Контекст: "
        for i in best:
            self.rag += self._chunk_data[i] + " "

    def send_message(self, message):
        if self.data == None:
            self.rag=""
        else:
            self.__generate_message_RAG(message)
        load_dotenv()
        apiKey = os.getenv("API_KEY")

        client = openai.OpenAI(
            api_key=apiKey,
            base_url="https://api.intelligence.io.solutions/api/v1/",
        )

        responce = client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it",
            messages=[
                {"role": "system", "content": self.rag},
                {"role": "user", "content": message}
            ]
        )

        return [responce.choices[0].message.content, self.rag]