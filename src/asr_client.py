"""
Servidor WebSocket para tradução simultânea de fala usando SeamlessM4T v2.
Recebe chunks de áudio (PCM 16-bit) do cliente (navegador) via WebSocket,
processa com o modelo, e retorna o texto traduzido (pt -> en).

Para rodar:
    python -m http.server 8000
    python src/asr_client.py

O servidor escuta em todas as interfaces de rede (0.0.0.0) na porta 8080,
permitindo conexões de outros dispositivos na mesma rede Wi-Fi.
"""

import asyncio
import json
import numpy as np
import torch
import websockets
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

# ===== Configuração =====
MODEL_NAME = "facebook/seamless-m4t-v2-large"
SAMPLE_RATE = 16000  # taxa de amostragem esperada (Hz)
HOST = "0.0.0.0"     # escuta em todas as interfaces de rede
PORT = 8080

# ===== Carregar modelo (uma vez, na inicialização) =====
print("Carregando SeamlessM4T v2... (pode demorar na primeira vez)")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME).to(device)
print(f"Modelo carregado em: {device}")


def processar_audio(audio_int16_bytes: bytes) -> str:
    """
    Recebe bytes de áudio PCM 16-bit (mono) e retorna o texto traduzido (en).
    """
    # Converter bytes -> array numpy int16 -> float32 normalizado [-1, 1]
    audio_int16 = np.frombuffer(audio_int16_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    inputs = processor(
        audio=audio_float32,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_tokens = model.generate(**inputs, tgt_lang="eng")

    texto = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    return texto


async def handler(websocket):
    print(f"Cliente conectado: {websocket.remote_address}")
    buffer = bytearray()

    # Acumula ~3 segundos de áudio antes de processar (ajuste conforme necessário)
    CHUNK_THRESHOLD_BYTES = SAMPLE_RATE * 2 * 3  # 3s, 16-bit (2 bytes/amostra)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                buffer.extend(message)

                if len(buffer) >= CHUNK_THRESHOLD_BYTES:
                    try:
                        texto = processar_audio(bytes(buffer))
                        if texto.strip():
                            await websocket.send(json.dumps({
                                "type": "translation",
                                "text": texto
                            }))
                            print("Traduzido:", texto)
                    except Exception as e:
                        print("Erro ao processar áudio:", e)
                    finally:
                        buffer.clear()

            elif isinstance(message, str):
                # mensagens de controle (JSON), se precisar no futuro
                data = json.loads(message)
                print("Mensagem de controle recebida:", data)

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")


async def main():
    print(f"Servidor WebSocket rodando em ws://{HOST}:{PORT}")
    print("Use o IP local do seu PC para conectar do celular (mesma rede Wi-Fi).")
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        await asyncio.Future()  # roda para sempre


if __name__ == "__main__":
    asyncio.run(main())