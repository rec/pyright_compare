from __future__ import annotations

import argparse
import contextlib
import dataclasses as dc
import json
import re
import subprocess
import sys
from collections import Counter
from functools import cache, cached_property, partial, wraps
from pathlib import Path
from typing import Any, Callable, cast, Iterator, Sequence, TYPE_CHECKING, TypeVar
from typing_extensions import TypeAlias


if TYPE_CHECKING:
    from collections.abc import Iterable

_T = TypeVar("_T")

_print = partial(print, file=sys.stderr)

INDENT_JSON = 4
COMMIT_ID_LENGTH = 11  # Same as git log
MAX_ERROR_CHARS = 2_048
INTERRUPT_COUNT = 32
SCORE_NAME = "completenessScore"
PULL_NUMBER_RE = re.compile(r".*\(#(\d+)\).*")

DESCRIPTION = """`pyright_compare` reports on typing for function and method
return values and arguments."""


IntDict = dict[str, int]

Number = int | float
NumberOrDict: TypeAlias = Number | dict[str, Number]
NumberOrDictDict: TypeAlias = dict[str, NumberOrDict]
Numbers: TypeAlias = NumberOrDict | NumberOrDictDict

Op: TypeAlias = Callable[[Number, Number], Number]
Strs: TypeAlias = list[str]
AnyDict: TypeAlias = dict[str, Any]
AnyDictDict: TypeAlias = dict[str, AnyDict]

_BUILD_STUBS = "cmake --build . --target torch_python_stubs"
_GIT_COMMIT_MESSAGE = "git log --format=%B -n 1"
_GIT_COUNT = "git rev-list --count"
_GIT_RESET = "git -c submodule.recurse=false reset --hard"
_GIT_REV_PARSE = "git rev-parse"
_GIT_COMMITTER_DATE = "git show --no-patch --format=%ci"
_PYRIGHT = "pyright --verifytypes torch --ignoreexternal --outputjson"


def main() -> None:
    sys.exit(PyrightCompare().compare())


@dc.dataclass(frozen=True)
class Commit:
    name: str
    _comp: PyrightCompare

    def asdict(self) -> dict[str, Any]:
        fields = "committer_date", "commit_id", "message"
        return {k: getattr(self, k) for k in fields}

    @cached_property
    def committer_date(self) -> str:
        return self._comp._git_committer_date(self.name)

    @cached_property
    def commit_id(self) -> str:
        return self._comp._git_commit_id(self.name)[:COMMIT_ID_LENGTH]

    @cached_property
    def message(self) -> str:
        return self._comp._git_commit_message(self.name).splitlines()[0]

    @cached_property
    def pyright(self) -> AnyDict | None:
        """Returns a dict with parsed JSON data, or None if the stubs build failed"""
        self._comp._git_reset(self.commit_id)
        if not self._comp._build_stubs():
            return None

        pyright = json.loads(self._comp._pyright() or "{}")
        assert isinstance(pyright, dict), pyright
        return pyright

    @cached_property
    def report(self) -> Report:
        return Report.create(self)


@dc.dataclass
class Report:
    """Extract information from `pyright --outputjson --verifytypes` into Python classes

    https://docs.basedpyright.com/latest/configuration/command-line/#json-output is the
    input format expected, though in fact pyright sometimes includes another
    undocumented field, "alternateNames".
    """

    commit: Commit
    numbers: NumberOrDictDict
    symbols: AnyDictDict

    @staticmethod
    def create(commit: Commit) -> Report:
        assert commit.pyright, f"{commit} failed to build"
        tc = commit.pyright["typeCompleteness"]

        filesAnalyzed = commit.pyright["summary"]["filesAnalyzed"]
        numbers = dict(_get_numbers(tc), filesAnalyzed=filesAnalyzed)

        counter = Counter(s["name"] for s in tc["symbols"])
        if dupes := [k for k, v in counter.items() if v > 1]:
            raise ValueError(f"{dupes=}")

        symbols = {s["name"]: s for s in tc["symbols"]}
        return Report(commit, numbers=numbers, symbols=symbols)

    def operate(self, other: Report, op: Op) -> Numbers:
        return _operate(op, self.numbers, other.numbers)


@dc.dataclass
class StringsDiff:
    added: Strs
    common: Strs
    removed: Strs

    @staticmethod
    def create(a: Iterable[str], b: Iterable[str]) -> StringsDiff:
        sa, sb = set(a), set(b)
        return StringsDiff(
            added=sorted(sb - sa), removed=sorted(sa - sb), common=sorted(sa & sb)
        )

    def add_remove(self) -> dict[str, Strs]:
        return {"added": self.added, "removed": self.removed}


@dc.dataclass
class SymbolsDiff:
    """A diff between two dicts of "symbols" - the dicts we get back
    from pyright in the "symbol" field."""

    added: Strs
    common: AnyDictDict
    removed: Strs

    @staticmethod
    def create(a: AnyDictDict, b: AnyDictDict) -> SymbolsDiff:
        diff = StringsDiff.create(a, b)
        return SymbolsDiff(
            common={k: v for k in diff.common if (v := _diff_symbol(a[k], b[k]))},
            **diff.add_remove(),
        )


@dc.dataclass
class ReportDiff:
    absolute: Numbers
    percent: Numbers
    symbols: SymbolsDiff
    completeness: list[Number]

    @staticmethod
    def create(a: Report, b: Report) -> ReportDiff:
        ca, cb = a.numbers["completenessScore"], b.numbers["completenessScore"]
        assert isinstance(ca, float) and isinstance(cb, float)
        return ReportDiff(
            absolute=b.operate(a, _sub),
            completeness=[ca, cb],
            percent=a.operate(b, _percent),
            symbols=SymbolsDiff.create(a.symbols, b.symbols),
        )


@dc.dataclass
class Summary:
    committer_date: str
    commit_id: str
    index: int
    message: str
    filename: filename
    completeness: float
    diagnostics: IntDict
    diff: AnyDict
    symbols: IntDict

    def asdict(self) -> AnyDict:
        d = dc.asdict(self)
        if not any(self.diagnostics.values()):
            d.pop("diagnostics")
        return d

    def as_markdown(self) -> str:
        date = self.committer_date.partition(" ")[0]
        url = f"https://github.com/pytorch/pytorch/commit/{self.commit_id}"
        delta = self.completeness_delta
        report = f"diffs/{self.filename}"  # TODO
        return f"* `{date}`: [`{delta:+.3f}%`]({report}) [`{self.message}`]({url})"

    @property
    def completeness_delta(self) -> float:
        return 100 * self.diff.get("completenessScore", 0.0)

    @staticmethod
    def create(filename: Path) -> Summary:
        *_, index, commits = filename.stem.split(".")
        commit_id = commits.split("-")[-1]

        data = json.loads(filename.read_text())
        absolute_diff = data["diff"].get("absolute", {})
        symbols = data["diff"].get("symbols", {})
        common = symbols.get("common", {}).values()

        after = data["commits"]["after"]
        committer_date, message = after["committer_date"], after["message"]
        completeness = 100 * data["completeness"][1]

        added_removed = "added", "removed"

        def to_length(d: dict[str, Any]) -> IntDict:
            return {k: len(v) for k in added_removed if (v := d.get(k))}

        it = (d for v in common if (d := to_length(v.get("diagnostics", {}))))
        diagnostics = {k: sum(d.get(k, 0) for d in it) for k in added_removed}

        return Summary(
            commit_id=commit_id,
            committer_date=committer_date,
            completeness=completeness,
            diagnostics=diagnostics,
            diff=absolute_diff,
            filename=data["filename"],
            index=int(index),
            message=message,
            symbols=to_length(symbols),
        )


class PyrightCompare:
    def compare(self) -> str | int:
        """Returns an error string, or 0 on success"""
        try:
            if self.args.summarize_json or self.args.summarize_markdown:
                self._summarize()
            else:
                with self._git_back_to_head():
                    self._compare()
            return 0
        except KeyboardInterrupt:
            return "KeyboardInterrupt"
        except Exception as e:
            error = " ".join(["ERROR: ", *(str(i) for i in e.args)])
            if isinstance(e, CompareError):
                return error
            else:
                _print(error)
                raise

    def _compare(self) -> None:
        if self.N < 1:
            msg = f"Need at least two commits to compare, have {len(self.commit_ids)}"
            raise CompareError(msg)

        if self.args.verbose:
            _print("Performing", self.N, "comparison" + "s" * (self.N != 1))

        # The loop goes backward from most recent to least recent commit.
        after = Commit("(not used)", self)
        for i, commit_id in enumerate(self.commit_ids):
            try:
                before = Commit(commit_id, self)
                if i:
                    self._compare_one(before, after, i - 1)
                after = before
            except Exception as e:
                b, a = before.name, after.name
                err = f"In comparing {b} and {a} ({i} out of {self.N}):"
                e.args = err, *e.args
                raise

    def _compare_one(self, before: Commit, after: Commit, index: int) -> None:
        errors = [c.commit_id for c in (after, before) if not c.pyright]
        if errors:
            if self.args.fail_if_build_fails:
                raise CompareError("Failed to build stubs for", *errors)
            else:
                _print("Failed to build", *errors)
                return

        if self.args.verbose:
            _print("Comparing", before.commit_id, "to", after.commit_id, "at", index)

        d = self._diff(before, after, index)
        prefix = f"{index:{self.digits}}:"
        path = self.output / d["filename"]
        if "absolute" in d["diff"] or self.args.write_empty:
            with path.open("w") as fp:
                fp.writelines(_dumps(d, indent=INDENT_JSON))
            _print(prefix, f"Wrote {path}")
        else:
            _print(prefix, f"Skipped empty {path}")

    def _diff(self, before: Commit, after: Commit, index: int) -> AnyDict:
        if self.args.index_invert:
            index = self.N - index - 1

        bc, ac = before.commit_id, after.commit_id
        filename_stem = self.args.filename_stem
        diff = {
            "commits": {"before": before.asdict(), "after": after.asdict()},
            "filename": f"{filename_stem}.{index:0{self.digits}}.{bc}-{ac}.json",
            "diff": dc.asdict(ReportDiff.create(before.report, after.report)),
        }
        return diff if self.args.keep_empty else _clean(diff)

    def _run(self, *cmds: str, ignore_errors: bool = False, **kwargs: Any) -> str:
        cmd = " ".join(cmds).split()  # No need for shlex yet
        if self.args.verbose:
            print("$", *cmd)
        cp = subprocess.run(cmd, text=True, capture_output=True, **kwargs)

        if self.args.verbose or (not ignore_errors and cp.returncode):
            _print(cp.stdout[:MAX_ERROR_CHARS])
            _print(cp.stderr if cp.returncode else cp.stderr[:MAX_ERROR_CHARS])

        if not ignore_errors and cp.returncode:
            raise CompareError(f"Command '{' '.join(cmds)}' failed")

        assert isinstance(cp.stdout, str)
        return cp.stdout.strip()

    def _summarize(self) -> None:
        glob = f"{self.args.filename_stem}.*.*-*.json"
        if not (files := sorted(self.output.glob(glob))):
            raise CompareError(f"No .json files in {self.output}")

        summaries = [Summary.create(f) for f in files]
        if not self.args.keep_reverts:
            def key(s: Summary) -> str:
                m = PULL_NUMBER_RE.match(s.message)
                return m.group(1) if m else s.commit_id

            deduped = {key(s): s for s in summaries}.values()
            summaries = [s for s in deduped if not s.message.startswith("Revert \"")]

        if self.args.sort_by_score:
            summaries.sort(key=lambda s: s.completeness_delta)

        if self.args.summarize_markdown:
            for s in summaries:
                print(s.as_markdown())
        else:
            sys.stdout.writelines(_dumps([s.asdict() for s in summaries], indent=4))

    @cached_property
    def args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        help = "The last commit in the series"
        parser.add_argument("last_commit", default="HEAD", nargs="?", help=help)

        help = "The first commit to start from"
        parser.add_argument("first_commit", default="", nargs="?", help=help)

        help = "Range  of commits to compare. None means 'cover first to last'"
        parser.add_argument("-c", "--commit-range", type=int, default=0, help=help)

        help = (
            "Minimum number of digits for the index number in the file name."
            " 0 means automatic."
        )
        parser.add_argument("-d", "--index-digits", type=int, default=0, help=help)

        help = "Filename stem for output files"
        parser.add_argument(
            "-f", "--filename-stem", default="pyright_compare", help=help
        )

        help = "Fail if the stubs build fails, otherwise skip over that commit"
        parser.add_argument("--fail-if-build-fails", action="store_true", help=help)

        help = "Count index numbers backwards so they decrease as time goes on"
        parser.add_argument("-i", "--index-invert", action="store_true", help=help)

        help = "Summarize the output directory into JSON"
        parser.add_argument("-j", "--summarize-json", action="store_true", help=help)

        help = "Keep empty diffs (zeroes and empty lists)"
        parser.add_argument("-k", "--keep-empty", action="store_true", help=help)

        help = "Keep reverts and resubmissions of the same pull request in summaries"
        parser.add_argument("--keep-reverts", action="store_true", help=help)

        help = "Summarize the output directory into markdown"
        parser.add_argument("-m", "--summarize-markdown", action="store_true", help=help)

        help = "Output directory for the diff files"
        parser.add_argument("-o", "--output", default="pyright_compare", help=help)

        help = "Sort summary output by score"
        parser.add_argument("-r", "--sort-by-score", action="store_true", help=help)

        help = (
            "How many commits to step by. None means 'compare first and last commit'"
            " if `first_commit` is set, else 1"
        )
        parser.add_argument("-s", "--step", default=None, type=int, help=help)

        help = (
            "Offset for the numerical index in filenames, used to continue a previous run"
            " from where it left off"
        )
        parser.add_argument("-t", "--index-start", type=int, default=0, help=help)

        help = "Print more debug info"
        parser.add_argument("-v", "--verbose", action="store_true", help=help)

        help = "Write files even when there are no type diffs"
        parser.add_argument("-w", "--write-empty", action="store_true", help=help)

        return parser.parse_args()

    @cached_property
    def commit_ids(self) -> Sequence[str]:
        commit_id = self._git_commit_id(self.args.last_commit)
        return [f"{commit_id}~{i}" for i in self.commit_indexes]

    @cached_property
    def commit_indexes(self) -> range:
        return range(self.start, self.stop, self.step)

    @cached_property
    def digits(self) -> int:
        return self.args.index_digits or len(str(self.commit_indexes.stop))

    @cached_property
    def N(self) -> int:
        """Number of diffs"""
        return len(self.commit_ids) - 1

    @cached_property
    def output(self) -> Path:
        if not (output := Path(self.args.output)).exists():
            if self.args.verbose:
                _print("Creating directory", output)
            output.mkdir(parents=True)
        return output

    @cached_property
    def start(self) -> int:
        assert isinstance(self.args.index_start, int)
        if self.args.index_start >= self.stop:
            raise CompareError(f"--index-start={self.args.index_start} >= {self.stop=}")
        return self.args.index_start

    @cached_property
    def step(self) -> int:
        return self.args.step or (1 if self.args.commit_range else self.stop - 1)

    @cached_property
    def stop(self) -> int:
        first, last = self.args.first_commit, self.args.last_commit
        if not first:
            end = self.args.commit_range
        elif self.args.commit_range:
            raise CompareError("--commit-range/-c and first_commit are both set")
        elif not (end := self._git_count(first, last)):
            raise CompareError(f"{first} isn't an ancestor of {last}")
        return 1 + (end or self.args.step or 1)

    def _build_stubs(self) -> bool:
        try:
            self._run(_BUILD_STUBS, cwd="build")
            return True
        except CompareError:
            # The build fails for some intermediate commits, like 3a5677a380c
            return False

    def _git_committer_date(self, ref: str) -> str:
        return self._run(_GIT_COMMITTER_DATE, ref).partition("+")[0].strip()

    def _git_commit_id(self, ref: str) -> str:
        return self._run(_GIT_REV_PARSE, ref).strip()[:COMMIT_ID_LENGTH]

    def _git_commit_message(self, ref: str) -> str:
        return self._run(_GIT_COMMIT_MESSAGE, ref).strip()

    def _git_count(self, child: str, parent: str) -> int:
        return int(self._run(f"{_GIT_COUNT} {parent}..{child}"))

    def _git_reset(self, ref: str) -> None:
        self._run(_GIT_RESET, ref)

    @contextlib.contextmanager
    def _git_back_to_head(self) -> Iterator[None]:
        head = self._git_commit_id("HEAD")
        try:
            yield
        finally:
            for _ in range(INTERRUPT_COUNT):
                with contextlib.suppress(KeyboardInterrupt):
                    _print("\nRestoring HEAD, wait a moment...")
                    self._git_reset(head)
                    break

    def _pyright(self) -> str:
        # pyright always seems to return errorcode = 1 :-/ so we ignore it
        return self._run(_PYRIGHT, ignore_errors=True)



class CompareError(ValueError):
    pass


def _clean(x: _T) -> _T:
    if isinstance(x, dict):
        d = {k: c for k, v in x.items() if (c := _clean(v))}
        return cast(_T, d)
    else:
        return x


def _diff_symbol(a: AnyDict, b: AnyDict) -> AnyDict:
    diff: AnyDict = {}

    for k, v in a.items():
        if k == "alternateNames" or v == (w := b[k]):
            continue
        elif isinstance(v, bool):
            diff[k] = w
        elif k == "diagnostics":
            s = StringsDiff.create((i["message"] for i in v), (i["message"] for i in w))
            if d := {k: v for k, v in s.add_remove().items() if v}:
                diff[k] = d

    return diff


@wraps(json.dumps)
def _dumps(
    obj: Any, *, indent: int | None = None, width: int = 88, **kwargs: Any
) -> Iterator[str]:
    """Multi-line JSON which fits lists and dicts of scalars onto one line"""
    delta = " " * (indent or 4)

    def dumps(x: Any, total_indent: str = "", offset: int = 0) -> Iterator[str]:
        i, w = 0, width - offset
        split = isinstance(x, (dict, list)) and any((i := i + s) > w for s in sizes(x))
        if not split:
            yield json.dumps(x)
            return
        if isinstance(x, dict):
            delimiters = "{}"
            items = iter(x.items())
            key_format = '"{key}": '.format
        elif isinstance(x, list):
            delimiters = "[]"
            items = enumerate(x)
            key_format = "".format

        yield delimiters[0]
        yield "\n"

        next_indent = total_indent + delta
        for i, (k, v) in enumerate(items):
            yield next_indent
            yield (key := key_format(key=k))
            yield from dumps(v, next_indent, offset + len(key))
            if i != len(x) - 1:
                yield ","
            yield "\n"

        yield total_indent
        yield delimiters[1]
        if not total_indent:
            yield "\n"

    def sizes(x: Any) -> Iterator[int]:
        if isinstance(x, (dict, list)):
            it = (j for i in x.items() for j in i) if isinstance(x, dict) else iter(x)
            for i, v in enumerate(it):
                yield (i != 0) * 2  # The separator
                yield from sizes(v)
        elif isinstance(x, str):
            yield 2 + len(x)
        else:
            yield len(str(x))

    yield from dumps(obj)


def _get_numbers(d: AnyDict) -> NumberOrDictDict:
    def is_number_field(v: Any) -> bool:
        if isinstance(v, dict):
            return bool(v) and all(is_number_field(i) for i in v.values())
        elif isinstance(v, list):
            return bool(v) and all(is_number_field(i) for i in v)
        else:
            return isinstance(v, (int, float))

    return {k: v for k, v in d.items() if is_number_field(v)}


def _operate(op: Op, a: Numbers, b: Numbers, key: str = "") -> Numbers:
    if isinstance(a, dict):
        assert isinstance(b, dict), b
        assert a.keys() == b.keys(), (a, b)
        return {k: _operate(op, v, b[k], k) for k, v in a.items()}  # type: ignore[return-value]
    else:
        assert isinstance(a, (float, int)) and isinstance(b, (float, int)), (a, b, key)
        return op(a, b)


def _percent(a: Number, b: Number) -> Number:
    if a:
        return round(100 * (b - a) / a, 4)

    return float("inf") if b > 0 else 0 if b == 0 else -float("inf")


def _sub(a: Number, b: Number) -> Number:
    return round(a - b, 6)


if __name__ == "__main__":
    main()
