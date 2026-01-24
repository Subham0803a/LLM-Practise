from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from rich import print as rprint
import uvicorn
import asyncio
import json
import time
import os
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemini AI Assistant API")

# 1. Setup Client for 2026 Stable v1
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1")
)

MODEL_NAME = "gemini-2.5-flash"
SAVE_FILE = "gemini_history.json"

class GeminiAssistant:
    def __init__(self):
        self.total_tokens = 0
        self.session_history = []
        self.chat_session = None 
        self.load_history()
    
    def load_history(self):
            if os.path.exists(SAVE_FILE):
                try:
                    with open(SAVE_FILE, "r") as f:
                        data = json.load(f)
                        self.session_history = [types.Content(**m) for m in data.get("messages", [])]
                        self.total_tokens = data.get("total_tokens", 0)
                    print(f"✅ Loaded {len(self.session_history)} messages from history.")
                except Exception as e:
                    print(f"⚠️ History load failed: {e}")
    
    async def _ensure_session(self):
        """Creates the ASYNC session only when needed."""
        if self.chat_session is None:
            # We MUST use client.aio for the chat to be awaitable
            self.chat_session = client.aio.chats.create(
                model=MODEL_NAME,
                history=self.session_history
            )
            
    async def _ensure_session(self):
            """Creates the ASYNC session only when needed."""
            if self.chat_session is None:
                # We MUST use client.aio for the chat to be awaitable
                self.chat_session = client.aio.chats.create(
                    model=MODEL_NAME,
                    history=self.session_history
                )

    def save_history(self):
        """Saves current history to JSON file."""
        if self.chat_session:
            current_history = self.chat_session.get_history()
            serializable_history = [m.model_dump() for m in current_history]
            
            with open(SAVE_FILE, "w") as f:
                json.dump({
                    "messages": serializable_history,
                    "total_tokens": self.total_tokens
                }, f, indent=4)
            print(f"💾 History saved to {SAVE_FILE}")

    async def send_message(self, user_text: str):
        # 1. Initialize async session if it's the first message
        await self._ensure_session()
        
        start_time = time.perf_counter()
        try:
            # 2. Call the async send_message
            response = await self.chat_session.send_message(user_text)
            
            duration = time.perf_counter() - start_time
            usage = response.usage_metadata
            self.total_tokens += usage.total_token_count
            
            return {
                "text": response.text,
                "tokens": usage.total_token_count,
                "time": duration
            }
        except Exception as e:
            print(f"❌ SDK Error: {e}")
            return {"error": str(e)}


# Initialize the assistant once globally
bot = GeminiAssistant()

class ChatRequest(BaseModel):
    message: str
    
@app.get("/")
def read_root():
    return {"status": "online", "tokens_processed": bot.total_tokens}
    
@app.post("/chat")
async def chat_with_gemini(request: ChatRequest, background_tasks: BackgroundTasks):
    # 1. Get response from bot
    result = await bot.send_message(request.message)
    rprint(result)
    
    # 2. Check for errors from the SDK
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # 3. Save to file in the background (prevents lag)
    background_tasks.add_task(bot.save_history)
    
    return {
        "reply": result["text"],
        "stats": {
            "tokens": result["tokens"],
            "time": f"{result['time']:.2f}s"
        },
        "session": {
            "total_tokens": bot.total_tokens,
            "history_count": len(bot.session_history)
        }
    }
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)