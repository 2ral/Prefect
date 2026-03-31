from ...utils.ccov import extract_functions
from .grammar import RE2_GRAMMAR

# modelizer imports
from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer, convert_and_validate_ebnf_grammar
from modelizer.configs import SEED

# other imports
from pathlib import Path
from typing import Optional
from os.path import normpath, dirname, abspath, exists
from os import setsid, killpg, getpgid
import signal
import subprocess
import re


# subject class
class Re2_Subject(BaseSubject):
    def __init__(self, timeout: Optional[int] = None, trials = None):
        # wrapper.cpp should be compiled at this point
        super().__init__(timeout, trials)
        file_directory = normpath(dirname(abspath(__file__)))
        self._re2_directory = file_directory + '/'
        self._re2_command = "("\
                         f"cd {self._re2_directory};"\
                         './wrapper "{}";'\
                         ")"
        self._gcovr_command = "("\
                              f"cd {self._re2_directory};"\
                              "gcovr --json coverage.json;"\
                              ")"

    def __getstate__(self):
        state = super().__getstate__()
        state["re2_command"] = self._re2_command
        state["gcovr_command"] = self._gcovr_command
        state["re2_directory"] = self._re2_directory
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._re2_command = state["re2_command"]
        self._gcovr_command = state["gcovr_command"]
        self._re2_directory = state["re2_directory"]

    def __execute__(self, data: str | list | tuple) -> str | list:
        self._input = data
        re2_args = self._re2_command
        re2_args = re2_args.format(self._input)
        gcovr_args = self._gcovr_command
        
        # run re2 wrapper command
        try:
            re2_proc = subprocess.Popen(re2_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, preexec_fn=setsid)
            stdout, stderr = re2_proc.communicate(timeout=self._timeout)
            # print(stdout)
            # print(stderr)
        except subprocess.TimeoutExpired:
            self._state = ExecutionState.TIMEOUT
            # kill process
            killpg(getpgid(re2_proc.pid), signal.SIGTERM)
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
        else:
            rc = re2_proc.returncode if re2_proc else None
            self._state = ExecutionState.PASS if rc == 0 else ExecutionState.FAIL
        
        # run gcovr command
        try:
            gcovr_proc = subprocess.Popen(gcovr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, preexec_fn=setsid)
            stdout, stderr = gcovr_proc.communicate(timeout=self._timeout)
            # print(stderr)
        except subprocess.TimeoutExpired:
            self._state = ExecutionState.TIMEOUT
            # kill process
            killpg(getpgid(gcovr_proc.pid), signal.SIGTERM)
        except Exception as e:
            self._state = ExecutionState.EXCEPTION

        # read coverage
        if exists(self._re2_directory + "/coverage.json"):
            try:
                # extract coverage from json
                raw_coverage = extract_functions(self._re2_directory + "/coverage.json")
                coverage = []
                for file in raw_coverage:
                    coverage.extend(file.tokens())
                if self._state == ExecutionState.TIMEOUT:
                    self._output = [str(self._state.value), "ERROR:_RE2_or_GCovr_Timeouted"] + coverage
                else:
                    self._output = [str(self._state.value)] + coverage
            except Exception as e:
                self._state = ExecutionState.EXCEPTION
                self._output = [str(self._state.value), "ERROR:_Unable_To_Read_Coverage_Report"]
        else:
            self._state = ExecutionState.EXCEPTION
            self._output = [str(self._state.value), "ERROR:_Coverage_Report_Not_Found"]

        # clean up and delete gcda files and coverage.json
        p = Path(self._re2_directory)
        # remove all .gcda files recursively
        for f in p.rglob("*.gcda"):
            if f.is_file() or f.is_symlink():
                f.unlink()
        # remove coverage.json
        cov = p / "coverage.json"
        if cov.exists():
            cov.unlink()
        
        # temp duplicate removal hotfix
        seen = set()
        result = []

        for item in self._output:
            if item not in seen:
                seen.add(item)
                result.append(item)
        self._output = result
        
        return self._output


# generator class
class Re2_Generator(GeneratorInterface):
    def __init__(self, subject:Re2_Subject, seed=SEED, logger=None, fuzzer_factory=None, min_nonterminals=1, max_nonterminals=24):
        super().__init__(source="re2", target="coverage", subject=subject, seed=seed, logger=logger)
        self._grammar = convert_and_validate_ebnf_grammar(RE2_GRAMMAR)
        if fuzzer_factory == None:
            self._fuzzer = GrammarCoverageFuzzer(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)
        else:
            self._fuzzer = fuzzer_factory(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)

    def generate(self):
        while True:
            try:
                tokens = self._fuzzer.fuzz()
                re.compile(tokens)
                break
            except re.error:
                wrong+=1
                continue
        return tokens
