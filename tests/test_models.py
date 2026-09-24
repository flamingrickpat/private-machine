"""Direct model-config tests. No server, model launch, or module thread."""
import os
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from runtime.models import (
    ConnectionConfig, LlmPreset, ModelConfig, load_config, resolve_connection,
)


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.example.yaml")

    def test_default_and_named_startup_forms(self):
        default = resolve_connection(self.config)
        self.assertEqual(default, resolve_connection(self.config, preset="Default"))
        self.assertEqual(default.model, "qwen")
        self.assertEqual(default.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(default, resolve_connection(self.config, preset=LlmPreset.UnitTestEvaluator))

    def test_explicit_form_needs_no_config(self):
        connection = resolve_connection(base_url="http://localhost:8081/v1/", model="other", api_key="marker-secret")
        self.assertEqual(connection.base_url, "http://localhost:8081/v1")
        self.assertEqual(connection.api_key.get_secret_value(), "marker-secret")

    def test_unknown_mixed_and_partial_forms_fail(self):
        with self.assertRaises(ValueError):
            resolve_connection(self.config, preset="Missing")
        with self.assertRaises(ValueError):
            resolve_connection(self.config, preset="Default", base_url="http://localhost:8080")
        with self.assertRaises(ValueError):
            resolve_connection(base_url="http://localhost:8080", model="qwen")
        with self.assertRaises(ValueError):
            resolve_connection()

    def test_environment_reference_resolves_once(self):
        name = "PM_W1_TEST_SECRET"
        previous = os.environ.get(name)
        try:
            os.environ[name] = "first-marker"
            config = ConnectionConfig(base_url="http://localhost:8080/v1", model="qwen", api_key_env=name)
            connection = config.resolve()
            os.environ[name] = "second-marker"
            self.assertEqual(connection.api_key.get_secret_value(), "first-marker")
            self.assertEqual(os.environ[name], "second-marker")
            del os.environ[name]
            with self.assertRaises(ValueError):
                config.resolve()
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_secret_redaction_and_exclusive_credentials(self):
        marker = "credential-marker-not-a-real-key"
        config = ConnectionConfig(base_url="http://localhost/v1", model="qwen", api_key=marker)
        for value in (config, config.resolve()):
            self.assertNotIn(marker, repr(value))
            self.assertNotIn(marker, str(value.model_dump()))
            self.assertNotIn(marker, value.model_dump_json())
        with self.assertRaises(ValidationError) as error:
            ConnectionConfig(base_url="http://localhost/v1", model="qwen", api_key=marker, api_key_env="KEY")
        self.assertNotIn(marker, str(error.exception))

    def test_config_requires_evaluator_and_exactly_one_embedding_service(self):
        values = self.config.model_dump()
        del values["chat"][LlmPreset.UnitTestEvaluator]
        with self.assertRaises(ValidationError):
            ModelConfig.model_validate(values)
        with self.assertRaises(ValidationError):
            ModelConfig.model_validate({**self.config.model_dump(), "embedding_overrides": {}})

    def test_relative_script_and_cache_paths_use_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            text = Path("config.example.yaml").read_text(encoding="utf-8")
            text = text.replace('startup_script: ""', 'startup_script: "start.ps1"')
            path.write_text(text, encoding="utf-8")
            config = load_config(path)
            self.assertEqual(resolve_connection(config).startup_script, Path(directory) / "start.ps1")
            self.assertEqual(config.embedding.cache_dir, Path(directory) / ".runtime" / "embeddings")

    def test_credentials_in_urls_are_rejected_without_echo(self):
        with self.assertRaises(ValidationError) as error:
            ConnectionConfig(base_url="http://user:private-marker@localhost/v1", model="qwen")
        self.assertNotIn("private-marker", str(error.exception))

    def test_invalid_yaml_never_echoes_a_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text('api_key: "private-marker\n', encoding="utf-8")
            with self.assertRaises(ValueError) as error:
                load_config(path)
            self.assertNotIn("private-marker", str(error.exception))


if __name__ == "__main__":
    unittest.main()
