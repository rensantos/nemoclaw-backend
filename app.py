from fastapi import FastAPI

from api import router


app = FastAPI(title="Nemoclaw Backend")
app.include_router(router)
