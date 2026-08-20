import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cost_tracker import get_model_call_counts, reset_model_call_counts
from evaluation.metric_violations import check_metric_violations
from evaluation.metrics import _strip_comments_for_codesim, compute_code_similarity, compute_ttest_violation
from evaluation.report import format_decimal, format_value
from evaluation.runner import PipelineRunner
from llms.claude_model import ClaudeModel
from llms.gemini_model import GeminiModel
from llms.mr_helper import MRHelper
from metamorphic.relations.mrs import MR19, _clean_library_name


class MRRegressionTests(unittest.TestCase):
    def test_empty_library_name_uses_fallback(self):
        self.assertEqual(_clean_library_name(""), "a suitable standard library module")

    @patch("metamorphic.relations.mrs.mr_llm_transform", return_value="")
    def test_mr19_empty_solve_response_returns_original(self, transform):
        original = "Original problem"

        self.assertEqual(MR19().transform(original), original)
        transform.assert_called_once()

    @patch(
        "metamorphic.relations.mrs.mr_llm_transform",
        side_effect=["```python\nprint('ok')\n```", ""],
    )
    def test_mr19_empty_explanation_returns_original(self, transform):
        original = "Original problem"

        self.assertEqual(MR19().transform(original), original)
        self.assertEqual(transform.call_count, 2)


class MetricRegressionTests(unittest.TestCase):
    @patch("evaluation.metrics._encode_with_transformers")
    def test_codesim_removes_comments_but_preserves_hashes_in_strings(self, encode):
        encode.side_effect = RuntimeError("stop after preprocessing")

        compute_code_similarity(
            'value = "# retained"  # removed\n',
            "value = '# also retained'\n# removed too\n",
        )

        code_a, code_b = encode.call_args.args
        self.assertIn('"# retained"', code_a)
        self.assertNotIn("# removed", code_a)
        self.assertIn("'# also retained'", code_b)
        self.assertNotIn("# removed too", code_b)
        self.assertEqual(
            _strip_comments_for_codesim("value = 1\n"),
            _strip_comments_for_codesim("# heading\nvalue = 1  # explanation\n"),
        )

    @patch("evaluation.metric_violations.load_repeat_baseline")
    def test_static_metric_error_sentinels_are_not_compared(self, load_baseline):
        load_baseline.return_value = {
            "question": {
                "ncloc": (10, 11, 12),
                "cyclomatic_complexity": (2.0, 2.5, 3.0),
            }
        }
        violations = check_metric_violations(
            "question",
            {"ncloc": -1, "cyclomatic_complexity": -2},
        )

        self.assertNotIn("ncloc", violations)
        self.assertNotIn("cyclomatic_complexity", violations)

    def test_one_sided_ttests_only_flag_the_configured_direction(self):
        baseline = [9, 10, 10, 10, 11]

        above = compute_ttest_violation(baseline, 12, alternative="less")
        below = compute_ttest_violation(baseline, 8, alternative="less")
        self.assertTrue(above["significant"])
        self.assertFalse(below["significant"])

        below = compute_ttest_violation(baseline, 8, alternative="greater")
        above = compute_ttest_violation(baseline, 12, alternative="greater")
        self.assertTrue(below["significant"])
        self.assertFalse(above["significant"])

    @patch("evaluation.metric_violations.load_repeat_baseline")
    def test_metric_tests_use_mixed_directions(self, load_baseline):
        samples = (9, 10, 10, 10, 11)
        load_baseline.return_value = {
            "question": {
                "ncloc": samples,
                "cyclomatic_complexity": samples,
                "llm_call_time": samples,
                "total_tokens": samples,
                "token_efficiency": samples,
                "code_sim": samples,
            }
        }

        violations = check_metric_violations(
            "question",
            {
                "ncloc": 8,
                "cyclomatic_complexity": 8,
                "llm_call_time": 8,
                "total_tokens": 8,
                "token_efficiency": 12,
                "code_sim": 12,
            },
        )

        self.assertEqual(
            set(violations),
            {"ncloc", "cyclomatic_complexity", "total_tokens"},
        )

        violations = check_metric_violations(
            "question",
            {
                "llm_call_time": 12,
                "token_efficiency": 8,
                "code_sim": 8,
            },
        )
        self.assertEqual(
            set(violations),
            {"llm_call_time", "token_efficiency", "code_sim"},
        )

    def test_mr_runs_append_and_reuse_shared_base_results(self):
        reset_model_call_counts()
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return f"response-{len(calls)}"

        model = SimpleNamespace(
            model_name="test-model",
            generate=generate,
            last_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            last_call_time=0.1,
            last_finish_reason="stop",
        )
        samples = [
            {
                "question_id": qid,
                "question_title": qid,
                "question_content": "Solve it.",
                "difficulty": "easy",
                "platform": "test",
            }
            for qid in ("q1", "q2")
        ]

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def result_path(_model_name, relation_name=None, relation_id=None):
                if relation_name:
                    return root / f"mr-{relation_id}.json"
                return root / "base.json"

            relation_1 = SimpleNamespace(name="first", target="PD", transform=lambda question: question)
            relation_2 = SimpleNamespace(name="second", target="PD", transform=lambda question: question)

            with patch("evaluation.runner.build_results_path", side_effect=result_path), patch("builtins.print") as output:
                PipelineRunner([model], samples[:1], delay=0, relation=relation_1, relation_id=1).run()
                PipelineRunner([model], samples, delay=0, relation=relation_2, relation_id=2).run()

            base_entries = json.loads((root / "base.json").read_text(encoding="utf-8"))
            second_mr = json.loads((root / "mr-2.json").read_text(encoding="utf-8"))
            printed = " ".join(" ".join(map(str, call.args)) for call in output.call_args_list)

        self.assertEqual(len(calls), 5)
        self.assertEqual(get_model_call_counts(), {"main": 5, "helper": 0})
        self.assertEqual([entry["question_id"] for entry in base_entries], ["q1", "q2"])
        self.assertEqual(base_entries[0]["response"], "response-1")
        self.assertEqual(second_mr[0]["response"], "response-1")
        self.assertEqual(second_mr[1]["response"], "response-4")
        self.assertIn("base exists", printed)
        self.assertIn("base generated", printed)

    def test_static_metric_error_sentinels_display_as_unavailable(self):
        for sentinel in (-1, -1.0, -2, -2.0):
            with self.subTest(sentinel=sentinel):
                self.assertEqual(format_value(sentinel), "N/A")
                self.assertEqual(format_decimal(sentinel), "N/A")


class ProviderMetadataTests(unittest.TestCase):
    @patch("llms.mr_helper.update_costs")
    def test_helper_usage_is_sent_to_cost_tracker(self, update_costs):
        reset_model_call_counts()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="answer"),
            )],
            usage=usage,
        )
        helper = MRHelper.__new__(MRHelper)
        helper.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response),
            ),
        )
        helper._model_name = "helper-model"
        helper._last_usage = None

        with patch("llms.mr_helper._log_helper_call"):
            self.assertEqual(helper.transform("system", "user"), "answer")

        update_costs.assert_called_once_with(
            "helper-model",
            {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        self.assertEqual(get_model_call_counts(), {"main": 0, "helper": 1})

    @patch("llms.base.update_costs")
    def test_claude_records_timing_and_stop_reason(self, _update_costs):
        message = SimpleNamespace(
            content=[SimpleNamespace(text="answer")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )
        model = ClaudeModel.__new__(ClaudeModel)
        model._model_name = "claude-test"
        model.client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: message),
        )

        self.assertEqual(model.generate("prompt"), "answer")
        self.assertIsInstance(model.last_call_time, float)
        self.assertEqual(model.last_finish_reason, "end_turn")

    @patch("llms.base.update_costs")
    def test_gemini_records_timing_and_finish_reason(self, _update_costs):
        finish_reason = SimpleNamespace(name="MAX_TOKENS")
        response = SimpleNamespace(
            text="answer",
            candidates=[SimpleNamespace(finish_reason=finish_reason)],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=4,
                total_token_count=14,
            ),
        )
        model = GeminiModel.__new__(GeminiModel)
        model._model_name = "gemini-test"
        model.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kwargs: response),
        )

        self.assertEqual(model.generate("prompt"), "answer")
        self.assertIsInstance(model.last_call_time, float)
        self.assertEqual(model.last_finish_reason, "MAX_TOKENS")

    def test_runner_collects_finish_reason(self):
        model = SimpleNamespace(
            last_usage={"prompt_tokens": 1, "completion_tokens": 1},
            last_call_time=0.25,
            last_finish_reason="stop",
            generate=lambda prompt: "answer",
        )
        runner = PipelineRunner([], [], delay=0)
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "thinking_tokens": 0,
            "reported": False,
        }

        generated = runner._generate_responses(model, "prompt", totals)

        self.assertEqual(generated.primary_finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
