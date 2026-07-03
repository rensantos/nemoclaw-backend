from app import app


if __name__ == "__main__":
    import uvicorn

    from config import config

    uvicorn.run(app, host=config.backend.host, port=config.backend.port)
