# Future Tasks

## Product Direction

- Treat Nemoclaw Backend as the reusable inference management backend, not only
  as a Transformers server.
- Keep Nemoclaw Core focused on agents, memory, planning, skills, RAG, research
  workflows, and orchestration.
- Keep LLM provider support, engine integration, model listing, model
  selection, benchmarking, and GPU/runtime inspection inside Nemoclaw Backend.
- Do not duplicate backend-owned model catalogs, benchmark commands, provider
  clients, or GPU/runtime inspection logic in Nemoclaw Core.

## Operational Follow-up

- Verify `./scripts/start.sh` on the UBI machine inside the `llm` Conda env.
- Verify `/health`, `/v1/models`, and `/v1/chat/completions` with the configured
  model loaded on the RTX A4000.
- Verify the Typer CLI on the UBI machine, especially `backend start`,
  `backend status`, `backend health`, and `backend logs`.
- Verify `backend status` against the existing development launcher and the
  future systemd service once Phase 8 exists.
- Decide when to remove the temporary shell wrappers after CLI usage settles.
- Verify `backend model use` on the UBI machine and restart the backend to
  confirm the selected model is loaded at process start.
- Verify `backend gpu list`, `backend gpu current`, and `backend gpu monitor`
  on the UBI machine with the RTX A4000.
- Verify `backend benchmark latency`, `backend benchmark throughput`, and
  `backend benchmark vram` against the running UBI backend after the model is
  loaded.

## API Follow-up

- Add clearer validation for unsupported request fields if Nemoclaw clients
  start sending more OpenAI parameters.
- Consider implementing streaming later if the client needs token-by-token
  responses.
- Add response timing metadata only if it remains outside the OpenAI-compatible
  response body or is explicitly accepted by clients.
- Add future engines behind `InferenceEngine` only when a phase explicitly calls
  for them. Do not change `api.py` or Nemoclaw Core for engine-specific work.
- Future `OllamaEngine`, `VLLMEngine`, `LlamaCppEngine`, and
  `OpenAICompatibleEngine` support belongs inside Nemoclaw Backend.
- Do not add Ollama, vLLM, llama.cpp, OpenAI-compatible provider clients, or any
  other new engine until an explicit implementation phase asks for it.
- Runtime model switching and model lifecycle commands remain future work.
- Use `docs/model-lifecycle-design.md` as the Phase 5 design reference before
  implementing `backend model load`, `backend model unload`, or `backend model
  switch`.
- Future `backend model load`, `backend model unload`, and `backend model switch`
  commands should build on `ModelManager` without moving inference logic into it.
- Implement real concurrent benchmark execution when needed. Phase 4 accepts
  `--concurrency` but still runs requests sequentially.
- Implement first-token latency only after streaming responses exist.
- GPU selection, multi-GPU scheduling, MIG support, CUDA affinity, and
  monitoring dashboards remain future work.

## Testing Follow-up

- Add lightweight unit tests for config precedence:
  environment variables over YAML over defaults.
- Add API tests with a mocked model runtime so endpoint response shapes can be
  checked without loading a GPU model.
- Add CLI integration tests in the `llm` Conda environment after Typer is
  installed there.
- Add a deployment smoke-test checklist for the UBI machine.
