from fastapi import FastAPI

from api import get_inference_service, router


app = FastAPI(title="Nemoclaw Backend")
app.include_router(router)


@app.on_event("startup")
def _build_inference_service() -> None:
    """Construct the inference service when the server starts.

    api.py builds it lazily so importing the module stays cheap and
    testable, but a server must still load its model up front rather than
    making the first caller pay for it - and the GPU busy-check has to run
    before anything is loaded to be meaningful at all.
    """
    get_inference_service()
