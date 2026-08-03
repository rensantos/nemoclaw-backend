import threading
import unittest
from pathlib import Path

from engines.base import (
    EngineUnavailableError,
    LifecycleNotSupportedError,
    ModelNotFoundError,
    ModelUnavailableError,
)
from services.inference import InferenceService
from services.lifecycle import (
    LifecycleConflictError,
    LifecycleState,
    LifecycleUnavailableError,
    StreamingNotSupportedError,
)


class FakeEngine:
    supports_runtime_lifecycle = True
    supports_streaming = False

    def __init__(self, model_id="fake-model", runtime_pids=()):
        self.loaded = False
        self.calls = []
        self.model_id = model_id
        self._runtime_pids = list(runtime_pids)
        self.vram_warning = None

    def runtime_pids(self):
        return self._runtime_pids

    def load_model(self, model_id=None):
        self.loaded = True
        if model_id is not None:
            self.model_id = model_id
        self.calls.append(("load_model", model_id))

    def unload_model(self):
        self.loaded = False
        self.calls.append("unload_model")

    def switch_model(self, model_id):
        self.calls.append(("switch_model", model_id))
        self.model_id = model_id

    def vram_warning_for(self, model_id):
        return self.vram_warning

    def health(self):
        self.calls.append("health")
        return {"status": "ok"}

    def list_models(self):
        self.calls.append("list_models")
        return {"object": "list", "data": []}

    def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
        self.calls.append(
            ("chat", messages, max_tokens, temperature, requested_model, think)
        )
        return {
            "content": "hello",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    def generate_text(self, prompt, max_new_tokens, temperature, think=None):
        self.calls.append(("generate_text", prompt, max_new_tokens, temperature, think))
        return {"model": "fake", "response": prompt}


class FakeGPUInfo:
    def __init__(self, index, name, memory_used_mib, memory_total_mib):
        self.index = index
        self.name = name
        self.memory_used_mib = memory_used_mib
        self.memory_total_mib = memory_total_mib


class FakeGPUManager:
    def __init__(self, busy=None):
        self._busy = busy or []
        self.own_pids_seen = None

    def busy_gpus(self, own_pids=None, **kwargs):
        # Recorded so a test can assert our own runtime is excluded.
        self.own_pids_seen = own_pids
        return self._busy


class InferenceServiceTests(unittest.TestCase):
    def test_service_loads_engine_once_on_init(self):
        engine = FakeEngine()

        InferenceService(engine)

        self.assertTrue(engine.loaded)
        self.assertEqual(engine.calls, [("load_model", None)])

    def test_no_warning_when_gpu_manager_omitted(self):
        with self.assertRaises(AssertionError):
            with self.assertLogs("services.inference", level="WARNING"):
                InferenceService(FakeEngine())

    def test_no_warning_when_configured_gpu_idle(self):
        with self.assertRaises(AssertionError):
            with self.assertLogs("services.inference", level="WARNING"):
                InferenceService(FakeEngine(), FakeGPUManager(busy=[]))

    def test_warns_when_configured_gpu_already_busy(self):
        busy_gpu = FakeGPUInfo("2", "RTX A4000", 7000, 16384)

        with self.assertLogs("services.inference", level="WARNING") as logs:
            InferenceService(FakeEngine(), FakeGPUManager(busy=[busy_gpu]))

        self.assertTrue(any("GPU 2" in message for message in logs.output))
        self.assertTrue(any("7000" in message for message in logs.output))

    def test_gpu_busy_check_runs_before_engine_load(self):
        engine = FakeEngine()
        busy_gpu = FakeGPUInfo("2", "RTX A4000", 7000, 16384)

        with self.assertLogs("services.inference", level="WARNING"):
            InferenceService(engine, FakeGPUManager(busy=[busy_gpu]))

        # load_model is still the only engine call - the check doesn't
        # touch the engine itself, just logs before loading proceeds.
        self.assertEqual(engine.calls, [("load_model", None)])

    def test_service_delegates_health_and_models(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        health = service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(service.list_models(), {"object": "list", "data": []})
        self.assertIn("health", engine.calls)
        self.assertIn("list_models", engine.calls)

    def test_default_lifecycle_state_is_ready(self):
        service = InferenceService(FakeEngine())

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_health_includes_lifecycle_state(self):
        service = InferenceService(FakeEngine())

        self.assertEqual(service.health()["lifecycle_state"], "ready")

    def test_health_preserves_existing_engine_fields(self):
        class RichFakeEngine(FakeEngine):
            def health(self):
                self.calls.append("health")
                return {
                    "status": "ok",
                    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "cuda": True,
                    "gpu": "RTX A4000",
                }

        service = InferenceService(RichFakeEngine())

        health = service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["model"], "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertTrue(health["cuda"])
        self.assertEqual(health["gpu"], "RTX A4000")
        self.assertEqual(health["lifecycle_state"], "ready")

    def test_health_reports_loaded_and_target_model(self):
        service = InferenceService(FakeEngine("fake-model"))

        health = service.health()

        self.assertEqual(health["loaded_model"], "fake-model")
        self.assertIsNone(health["target_model"])

    def test_service_delegates_chat(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        result = service.chat(["message"], 32, 0.5)

        self.assertEqual(result["content"], "hello")
        self.assertIn(("chat", ["message"], 32, 0.5, None, None), engine.calls)

    def test_service_delegates_chat_with_requested_model(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        service.chat(["message"], 32, 0.5, requested_model="tiny")

        self.assertIn(("chat", ["message"], 32, 0.5, "tiny", None), engine.calls)

    def test_service_delegates_chat_with_think(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        service.chat(["message"], 32, 0.5, think=False)

        self.assertIn(("chat", ["message"], 32, 0.5, None, False), engine.calls)

    def test_service_delegates_generate_text(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        result = service.generate_text("prompt", 12, 0.7)

        self.assertEqual(result["response"], "prompt")
        self.assertIn(("generate_text", "prompt", 12, 0.7, None), engine.calls)

    def test_service_delegates_generate_text_with_think(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        service.generate_text("prompt", 12, 0.7, think=True)

        self.assertIn(("generate_text", "prompt", 12, 0.7, True), engine.calls)

    def test_chat_transitions_to_degraded_on_engine_unavailable(self):
        class UnavailableEngine(FakeEngine):
            def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
                raise EngineUnavailableError("daemon down")

        service = InferenceService(UnavailableEngine())

        with self.assertRaises(EngineUnavailableError):
            service.chat(["message"], 32, 0.5)

        self.assertEqual(service.lifecycle_state, LifecycleState.DEGRADED)

    def test_generate_text_transitions_to_degraded_on_engine_unavailable(self):
        class UnavailableEngine(FakeEngine):
            def generate_text(self, prompt, max_new_tokens, temperature, think=None):
                raise EngineUnavailableError("daemon down")

        service = InferenceService(UnavailableEngine())

        with self.assertRaises(EngineUnavailableError):
            service.generate_text("prompt", 12, 0.7)

        self.assertEqual(service.lifecycle_state, LifecycleState.DEGRADED)

    def test_chat_propagates_model_not_found_without_changing_lifecycle(self):
        class RejectingEngine(FakeEngine):
            def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
                raise ModelNotFoundError(requested_model, "servable-model")

        service = InferenceService(RejectingEngine())

        with self.assertRaises(ModelNotFoundError):
            service.chat(["message"], 32, 0.5, requested_model="bogus")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)


class FakeModelManager:
    """Stands in for ModelManager's configured-catalog gate."""

    def __init__(self, configured=("fake-model", "other-model")):
        self.configured = list(configured)
        self.selected = []

    def validate_model(self, model_id):
        if model_id not in self.configured:
            raise ValueError("Model is not configured: {}".format(model_id))

    def select_model(self, model_id):
        self.selected.append(model_id)


def _lifecycle_service(engine=None, model_manager=None):
    return InferenceService(
        engine or FakeEngine(),
        model_manager=model_manager or FakeModelManager(),
    )


class LifecycleTransitionTests(unittest.TestCase):
    def test_unload_then_load_walks_ready_unloaded_ready(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)

        service.unload_model()
        self.assertEqual(service.lifecycle_state, LifecycleState.UNLOADED)
        self.assertIsNone(service.loaded_model_id)

        result = service.load_model("fake-model")
        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(service.loaded_model_id, "fake-model")
        self.assertEqual(result["previous_model"], None)
        self.assertIn(("load_model", "fake-model"), engine.calls)

    def test_switch_goes_through_switching_and_delegates_to_engine(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)

        result = service.switch_model("other-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(service.loaded_model_id, "other-model")
        self.assertEqual(result["previous_model"], "fake-model")
        self.assertIn(("switch_model", "other-model"), engine.calls)

    def test_preflight_rejection_leaves_previous_model_serving(self):
        """Found live on UBI: switching to a configured-but-not-pulled tag
        left the backend degraded and refusing all inference, even though
        the engine had rejected the target before touching the loaded
        model and the old one was still fine.
        """
        class PreflightRejectingEngine(FakeEngine):
            def switch_model(self, model_id):
                raise ModelUnavailableError(
                    "Ollama tag '{}' is not present".format(model_id)
                )

        service = _lifecycle_service(PreflightRejectingEngine())

        with self.assertRaises(ModelUnavailableError):
            service.switch_model("other-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(service.loaded_model_id, "fake-model")
        self.assertIsNone(service.target_model_id)
        # Still serving, which is the whole point.
        self.assertEqual(service.chat(["message"], 32, 0.5)["content"], "hello")

    def test_failure_after_the_preflight_still_degrades(self):
        class MidTransitionFailureEngine(FakeEngine):
            def switch_model(self, model_id):
                raise RuntimeError("daemon died halfway through")

        service = _lifecycle_service(MidTransitionFailureEngine())

        with self.assertRaises(RuntimeError):
            service.switch_model("other-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.DEGRADED)

    def test_degraded_recovers_via_load(self):
        class FailOnceEngine(FakeEngine):
            fail = True

            def switch_model(self, model_id):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("boom")

        service = _lifecycle_service(FailOnceEngine())
        with self.assertRaises(RuntimeError):
            service.switch_model("other-model")

        service.load_model("other-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_unload_from_degraded_reports_unloaded(self):
        service = _lifecycle_service()
        service.lifecycle_state = LifecycleState.DEGRADED

        result = service.unload_model()

        self.assertEqual(service.lifecycle_state, LifecycleState.UNLOADED)
        self.assertEqual(result["loaded_model"], None)


class LifecycleGuardTests(unittest.TestCase):
    def test_load_same_model_is_idempotent(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)

        result = service.load_model("fake-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(result["loaded_model"], "fake-model")
        self.assertEqual(engine.calls, [("load_model", None)])

    def test_load_different_model_while_ready_is_a_conflict(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)

        with self.assertRaises(LifecycleConflictError) as ctx:
            service.load_model("other-model")

        self.assertIn("model switch", str(ctx.exception))
        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_unload_when_already_unloaded_is_idempotent(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)
        service.unload_model()
        engine.calls.clear()

        service.unload_model()

        self.assertEqual(engine.calls, [])
        self.assertEqual(service.lifecycle_state, LifecycleState.UNLOADED)

    def test_switch_when_not_ready_directs_to_load(self):
        service = _lifecycle_service()
        service.unload_model()

        with self.assertRaises(LifecycleConflictError) as ctx:
            service.switch_model("other-model")

        self.assertIn("model load", str(ctx.exception))

    def test_unconfigured_model_is_rejected_before_the_engine_is_touched(self):
        engine = FakeEngine()
        service = _lifecycle_service(engine)
        engine.calls.clear()

        with self.assertRaises(ValueError):
            service.switch_model("never-configured")

        self.assertEqual(engine.calls, [])
        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_engine_without_runtime_lifecycle_support_is_refused(self):
        class InProcessEngine(FakeEngine):
            supports_runtime_lifecycle = False

        service = _lifecycle_service(InProcessEngine())

        for operation in (
            lambda: service.load_model("other-model"),
            service.unload_model,
            lambda: service.switch_model("other-model"),
        ):
            with self.assertRaises(LifecycleNotSupportedError):
                operation()

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)


class LifecyclePersistenceTests(unittest.TestCase):
    def test_switch_does_not_persist_by_default(self):
        model_manager = FakeModelManager()
        service = _lifecycle_service(model_manager=model_manager)

        result = service.switch_model("other-model")

        self.assertEqual(model_manager.selected, [])
        self.assertFalse(result["persisted"])

    def test_switch_with_persist_writes_through_model_manager(self):
        model_manager = FakeModelManager()
        service = _lifecycle_service(model_manager=model_manager)

        result = service.switch_model("other-model", persist=True)

        self.assertEqual(model_manager.selected, ["other-model"])
        self.assertTrue(result["persisted"])


class LifecycleRequestRejectionTests(unittest.TestCase):
    def test_chat_and_generate_are_rejected_while_not_ready(self):
        for state in (
            LifecycleState.LOADING,
            LifecycleState.UNLOADING,
            LifecycleState.SWITCHING,
            LifecycleState.UNLOADED,
        ):
            service = _lifecycle_service()
            service.lifecycle_state = state

            with self.assertRaises(LifecycleUnavailableError):
                service.chat(["message"], 32, 0.5)
            with self.assertRaises(LifecycleUnavailableError):
                service.generate_text("prompt", 32, 0.5)

    def test_transition_drains_in_flight_requests_before_calling_the_engine(self):
        release = threading.Event()
        entered = threading.Event()

        class SlowEngine(FakeEngine):
            def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
                entered.set()
                release.wait(5)
                return super().chat(messages, max_tokens, temperature, requested_model, think)

        engine = SlowEngine()
        service = _lifecycle_service(engine)

        chat_thread = threading.Thread(
            target=lambda: service.chat(["message"], 32, 0.5)
        )
        chat_thread.start()
        self.assertTrue(entered.wait(5))

        switch_done = threading.Event()

        def do_switch():
            service.switch_model("other-model")
            switch_done.set()

        switch_thread = threading.Thread(target=do_switch)
        switch_thread.start()

        # The switch must still be blocked draining the in-flight chat.
        self.assertFalse(switch_done.wait(0.3))
        self.assertNotIn(("switch_model", "other-model"), engine.calls)

        release.set()
        chat_thread.join(5)
        switch_thread.join(5)

        self.assertTrue(switch_done.is_set())
        self.assertIn(("switch_model", "other-model"), engine.calls)

    def test_drain_proceeds_after_timeout(self):
        release = threading.Event()
        entered = threading.Event()

        class SlowEngine(FakeEngine):
            def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
                entered.set()
                release.wait(5)
                return super().chat(messages, max_tokens, temperature, requested_model, think)

        engine = SlowEngine()
        service = _lifecycle_service(engine)

        chat_thread = threading.Thread(target=lambda: service.chat(["m"], 32, 0.5))
        chat_thread.start()
        self.assertTrue(entered.wait(5))

        with self.assertLogs("services.inference", level="WARNING") as logs:
            service.switch_model("other-model", timeout=0.2)

        self.assertTrue(any("Drain timed out" in message for message in logs.output))
        release.set()
        chat_thread.join(5)


class ApiBoundaryTests(unittest.TestCase):
    """The one assertion that genuinely belongs at source level: the API
    layer must not reach into Transformers/CUDA internals. Everything else
    about the API is now tested behaviourally in test_api.py."""

    def test_api_layer_does_not_import_transformers_or_torch(self):
        api_source = Path("api.py").read_text(encoding="utf-8")

        self.assertNotIn("transformers", api_source)
        self.assertNotIn("import torch", api_source)


class VRAMWarningTests(unittest.TestCase):
    """The daemon's GPU set only changes on restart, so a frontend-driven
    switch can outgrow it. Advisory only - it must never fail the switch."""

    def test_switch_surfaces_the_engine_warning_in_the_result(self):
        engine = FakeEngine()
        engine.vram_warning = "needs 40000 MiB, only 32234 MiB reachable"
        service = _lifecycle_service(engine)

        result = service.switch_model("other-model")

        self.assertEqual(result["warning"], engine.vram_warning)
        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(service.loaded_model_id, "other-model")

    def test_no_warning_key_when_the_model_fits(self):
        service = _lifecycle_service(FakeEngine())

        self.assertNotIn("warning", service.switch_model("other-model"))

    def test_a_failing_warning_check_never_breaks_the_switch(self):
        class ExplodingWarning(FakeEngine):
            def vram_warning_for(self, model_id):
                raise RuntimeError("nvidia-smi exploded")

        service = _lifecycle_service(ExplodingWarning())

        result = service.switch_model("other-model")

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertNotIn("warning", result)


class CatalogFakeModelManager(FakeModelManager):
    def __init__(self, entries):
        super().__init__([str(e["id"]) for e in entries])
        self.entries = entries

    def list_models(self):
        return self.entries


class ModelListingTests(unittest.TestCase):
    """/v1/models drives the frontend's model picker: every selectable
    model, flagged with whether it is actually usable."""

    def _entries(self):
        return [
            {"id": "qwen3:30b", "engine": "ollama"},
            {"id": "qwen3:1.7b", "engine": "ollama"},
            {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "engine": "transformers"},
        ]

    def _service(self, runtime_info=None):
        class CatalogEngine(FakeEngine):
            def model_runtime_info(self, model_ids):
                return runtime_info or {}

        engine = CatalogEngine("qwen3:30b")
        return InferenceService(
            engine, model_manager=CatalogFakeModelManager(self._entries())
        )

    def test_lists_the_whole_catalog_not_just_the_loaded_model(self):
        service = self._service()

        ids = [m["id"] for m in service.list_models()["data"]]

        self.assertIn("qwen3:30b", ids)
        self.assertIn("qwen3:1.7b", ids)

    def test_excludes_entries_for_other_engines(self):
        service = self._service()

        ids = [m["id"] for m in service.list_models()["data"]]

        # config.backend.engine is ollama in this repo's config.
        self.assertNotIn("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ids)

    def test_marks_exactly_the_loaded_model(self):
        service = self._service()

        loaded = [m["id"] for m in service.list_models()["data"] if m["loaded"]]

        self.assertEqual(loaded, ["qwen3:30b"])

    def test_merges_engine_runtime_flags(self):
        service = self._service({
            "qwen3:30b": {"pulled": True, "size_mib": 26545, "fits": True},
            "qwen3:1.7b": {"pulled": False},
        })

        by_id = {m["id"]: m for m in service.list_models()["data"]}

        self.assertTrue(by_id["qwen3:30b"]["pulled"])
        self.assertTrue(by_id["qwen3:30b"]["fits"])
        self.assertEqual(by_id["qwen3:30b"]["size_mib"], 26545)
        self.assertFalse(by_id["qwen3:1.7b"]["pulled"])
        # Unknown keys are omitted, never guessed.
        self.assertNotIn("fits", by_id["qwen3:1.7b"])

    def test_keeps_the_openai_compatible_fields(self):
        service = self._service()

        model = service.list_models()["data"][0]

        for field in ("id", "object", "created", "owned_by"):
            self.assertIn(field, model)
        self.assertEqual(model["object"], "model")
        self.assertEqual(service.list_models()["object"], "list")

    def test_catalog_still_returned_when_runtime_info_fails(self):
        class BrokenEngine(FakeEngine):
            def model_runtime_info(self, model_ids):
                raise RuntimeError("nvidia-smi exploded")

        service = InferenceService(
            BrokenEngine("qwen3:30b"),
            model_manager=CatalogFakeModelManager(self._entries()),
        )

        ids = [m["id"] for m in service.list_models()["data"]]

        self.assertIn("qwen3:30b", ids)

    def test_unreachable_engine_propagates_rather_than_lying(self):
        class DownEngine(FakeEngine):
            def model_runtime_info(self, model_ids):
                raise EngineUnavailableError("daemon down")

        service = InferenceService(
            DownEngine("qwen3:30b"),
            model_manager=CatalogFakeModelManager(self._entries()),
        )

        with self.assertRaises(EngineUnavailableError):
            service.list_models()

    def test_falls_back_to_the_engine_without_a_model_manager(self):
        service = InferenceService(FakeEngine())

        self.assertEqual(service.list_models(), {"object": "list", "data": []})


class StreamingTests(unittest.TestCase):
    """Streaming must count as an active request for the whole stream, and
    must reject before the response starts (a generator would defer the
    rejection past the point where a status code can still be set)."""

    class StreamingEngine(FakeEngine):
        supports_streaming = True

        def chat_stream(self, messages, max_tokens, temperature,
                        requested_model=None, think=None):
            self.calls.append(("chat_stream", requested_model, think))
            yield {"reasoning": "pondering"}
            yield {"content": "Lis"}
            yield {"content": "bon"}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 2,
                             "total_tokens": 3}}

    def test_yields_engine_deltas_in_order(self):
        service = _lifecycle_service(self.StreamingEngine())

        deltas = list(service.chat_stream(["m"], 32, 0.5))

        self.assertEqual(deltas[0], {"reasoning": "pondering"})
        self.assertEqual(deltas[1]["content"], "Lis")
        self.assertEqual(deltas[-1]["usage"]["total_tokens"], 3)

    def test_rejects_eagerly_when_the_engine_cannot_stream(self):
        service = _lifecycle_service(FakeEngine())  # supports_streaming False

        # Must raise on the call itself, not on first iteration.
        with self.assertRaises(StreamingNotSupportedError):
            service.chat_stream(["m"], 32, 0.5)

    def test_rejects_eagerly_when_not_ready(self):
        service = _lifecycle_service(self.StreamingEngine())
        service.lifecycle_state = LifecycleState.SWITCHING

        with self.assertRaises(LifecycleUnavailableError):
            service.chat_stream(["m"], 32, 0.5)

    def test_engine_model_rejection_surfaces_on_the_call_not_on_iteration(self):
        """The real OllamaEngine.chat_stream() validates the requested
        model before returning its iterator. The service must therefore
        call the engine eagerly - if it only did so from inside its own
        generator, the rejection would land after HTTP 200 was sent and
        the client would see a dead connection instead of a 404."""

        class ValidatingEngine(FakeEngine):
            supports_streaming = True

            def chat_stream(self, messages, max_tokens, temperature,
                            requested_model=None, think=None):
                if requested_model not in (None, "qwen3:30b"):
                    raise ModelNotFoundError(requested_model, "qwen3:30b")
                return iter([{"content": "ok"}])

        service = _lifecycle_service(ValidatingEngine())

        with self.assertRaises(ModelNotFoundError):
            service.chat_stream(["m"], 32, 0.5, requested_model="qwen3:32b")

        # The rejected request must not have taken a serving slot.
        self.assertEqual(service._active_requests, 0)

    def test_stream_counts_as_an_active_request_until_it_finishes(self):
        service = _lifecycle_service(self.StreamingEngine())

        stream = service.chat_stream(["m"], 32, 0.5)
        self.assertEqual(service._active_requests, 0)  # not started yet

        next(stream)
        self.assertEqual(service._active_requests, 1)  # slot held mid-stream

        list(stream)
        self.assertEqual(service._active_requests, 0)  # released at the end

    def test_a_transition_drains_an_in_flight_stream(self):
        """The lifecycle design requires streams to be drained, not cut
        off mid-response."""
        service = _lifecycle_service(self.StreamingEngine())

        stream = service.chat_stream(["m"], 32, 0.5)
        next(stream)

        switch_done = threading.Event()

        def do_switch():
            service.switch_model("other-model")
            switch_done.set()

        switch_thread = threading.Thread(target=do_switch)
        switch_thread.start()

        self.assertFalse(switch_done.wait(0.3))  # blocked draining the stream

        list(stream)
        switch_thread.join(5)
        self.assertTrue(switch_done.is_set())

    def test_engine_failure_mid_stream_degrades_the_service(self):
        class FailingStream(FakeEngine):
            supports_streaming = True

            def chat_stream(self, *args, **kwargs):
                yield {"content": "partial"}
                raise EngineUnavailableError("daemon died")

        service = _lifecycle_service(FailingStream())

        with self.assertRaises(EngineUnavailableError):
            list(service.chat_stream(["m"], 32, 0.5))

        self.assertEqual(service.lifecycle_state, LifecycleState.DEGRADED)
        self.assertEqual(service._active_requests, 0)  # slot still released
