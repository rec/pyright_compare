# mypy: ignore-errors

import dataclasses
import io
import json
import os
import sys
from pathlib import Path
from unittest import mock, TestCase

import pyright_compare as pc


TESTDATA = Path(__file__).parent / "testdata"
TEST_FILE1 = TESTDATA / "pyright1.json"
TEST_FILE2 = TESTDATA / "pyright2.json"
DIFF = TESTDATA / "pyright_compare.081045.47cdad29952-284b7668980.json"


class TestPyrightCompare(TestCase):
    maxDiff = 10_240
    rewrite_expected = "REWRITE_EXPECTED" in os.environ

    def assertExpected(self, path: Path, actual: str, suffix: str) -> None:
        # TODO: Simplify this method copied from elsewhere
        expected_file = Path(f"{path}.{suffix}")
        if not self.rewrite_expected and expected_file.exists():
            self.assertEqual(expected_file.read_text(), actual)
        else:
            with expected_file.open("w") as fp:
                fp.writelines(actual)

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("sys.argv", new=sys.argv[:1])
    def test_pyright(self, mock_stdout):
        compare = pc.PyrightCompare()
        c1, c2 = "dad54ca7c05", "3034dcc8032"
        compare.commit_ids = [c1, c2]

        def make_commit(name, commit_id, message, pyright_file):
            pyright = json.loads(pyright_file.read_text())
            c = pc.Commit(name, compare)
            c.__dict__.update(commit_id=commit_id, message=message, pyright=pyright)
            return c

        before = make_commit("HEAD~", c1, "A commit message", TEST_FILE1)
        after = make_commit("HEAD", c2, "A comet massage", TEST_FILE2)
        report_diff = compare._diff(before, after, 0)
        diff = _dumps(report_diff)
        self.assertExpected(TESTDATA / "diff", diff, "json")

    def test_summary(self):
        summary = dataclasses.asdict(pc.Summary.create(DIFF))
        self.assertExpected(TESTDATA / "summary", _dumps(summary), "json")

    def test_simple_json(self):
        actual = _dumps({"a": {"b": 1}})
        expected = '{"a": {"b": 1}}'
        self.assertEqual(actual, expected)

    def test_json(self):
        deep = json.loads(DEEP_STR)
        self.assertEqual(_dumps(deep), DEEP_STR)

        tests = {}, [], [deep, deep, deep], {"a": deep, "b": deep, "c": deep}
        for t in tests:
            self.assertEqual(json.loads(_dumps(t)), t)


def _dumps(x):
    return "".join(pc._dumps(x))


EXPECTED = """{
    "a": {"b": 1}
}
"""
DEEP_STR = """{
    "diff": {
        "exported": {"known": 7, "ambiguous": -2, "unknown": -4},
        "other": {"known": 11, "ambiguous": 2, "unknown": -7},
        "completenessscore": 0.0029010374569277686
    },
    "symbols": {"added": 2, "removed": 1},
    "diagnostics": {"added": 2, "removed": 1}
}
"""
