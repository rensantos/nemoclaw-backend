from services.inference import create_inference_service


inference_service = create_inference_service()


def health():
    return inference_service.health()


def list_models():
    return inference_service.list_models()


def generate_chat(messages, max_tokens, temperature):
    return inference_service.chat(messages, max_tokens, temperature)


def generate_text(prompt: str, max_new_tokens: int, temperature: float):
    return inference_service.generate_text(prompt, max_new_tokens, temperature)
