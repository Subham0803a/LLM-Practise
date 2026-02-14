from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import torch
import uvicorn

load_dotenv()

app = FastAPI(title="Prompting with LLMs")

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set pad token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# 1. Setup Client for 2026 Stable v1
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1")
)

MODEL_NAME = "gemini-2.5-flash"

class ZeroShotRequest(BaseModel):
    task: str  # summarization, translation, qa
    input_text: str
    
@app.get("/")
def read_root():
    return {"message": "welcome to prompting"}

@app.post("/zero-shot")
def zero_shot_gpt2(request: ZeroShotRequest):
    """
    Zero-shot prompting with GPT-2
    
    How it works:
    1. Create direct prompt (no examples)
    2. GPT-2 generates completion
    3. Return result
    
    Parameters:
    - task: Type of task (summarization, translation, qa)
    - input_text: The text to process
    """
    # Create zero-shot prompt based on task
    if request.task == "summarization":
        prompt = f"Summarize the following text:\n\n{request.input_text}\n\nSummary:"
    elif request.task == "translation":
        prompt = f"Translate to French: {request.input_text}\nFrench:"
    elif request.task == "qa":
        prompt = f"Answer this question: {request.input_text}\nAnswer:"
    else:
        prompt = request.input_text
    
    # Tokenize input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response with GPT-2
    with torch.no_grad():
        # max_length: Maximum tokens to generate
        # num_return_sequences: Number of completions
        # temperature: 0.7 for balanced creativity
        # do_sample: Enable sampling (vs greedy decoding)
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,  # Prompt length + 100 new tokens
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    result = generated_text[len(prompt):].strip()
    
    return {
        "task": request.task,
        "original_prompt": prompt,
        "zero_shot_result": result,
        "model": "GPT-2",
        "prompt_type": "Zero-shot (no examples provided)",
        "tokens_generated": len(outputs[0]) - len(inputs[0])
    }

@app.get("/zero-shot-examples")
def zero_shot_examples():
    """Example zero-shot prompts"""
    return {
        "summarization": {
            "prompt": "Summarize the following text:\n\nArtificial intelligence is transforming industries...\n\nSummary:",
            "note": "Direct task, no examples"
        },
        "translation": {
            "prompt": "Translate to French: Hello, how are you?\nFrench:",
            "note": "Direct translation request"
        },
        "qa": {
            "prompt": "Answer this question: What is the capital of France?\nAnswer:",
            "note": "Direct question"
        }
    }

if __name__ == "__main__":
    # The uvicorn.run command starts the server
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)