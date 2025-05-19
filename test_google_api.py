import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Chave API não encontrada. Verifique o .env")
else:
    genai.configure(api_key=API_KEY)
    print("Modelos disponíveis:")
    for m in genai.list_models():
        # Verificar se o modelo suporta 'generateContent' (para chat/texto)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- Nome: {m.name}, Display Name: {m.display_name}, Métodos: {m.supported_generation_methods}")