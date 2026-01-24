from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    # The uvicorn.run command starts the server
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)