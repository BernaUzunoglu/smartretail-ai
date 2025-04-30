import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", reload=True)
