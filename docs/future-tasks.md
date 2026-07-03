# Future Tasks

## Operational Follow-up

- Verify `./scripts/start.sh` on the UBI machine inside the `llm` Conda env.
- Verify `/health`, `/v1/models`, and `/v1/chat/completions` with the configured
  model loaded on the RTX A4000.
- Verify the Typer CLI on the UBI machine, especially `backend start`,
  `backend status`, `backend health`, and `backend logs`.
- Decide when to remove the temporary shell wrappers after CLI usage settles.

## API Follow-up

- Add clearer validation for unsupported request fields if Nemoclaw clients
  start sending more OpenAI parameters.
- Consider implementing streaming later if the client needs token-by-token
  responses.
- Add response timing metadata only if it remains outside the OpenAI-compatible
  response body or is explicitly accepted by clients.

## Testing Follow-up

- Add lightweight unit tests for config precedence:
  environment variables over YAML over defaults.
- Add API tests with a mocked model runtime so endpoint response shapes can be
  checked without loading a GPU model.
- Add a deployment smoke-test checklist for the UBI machine.
