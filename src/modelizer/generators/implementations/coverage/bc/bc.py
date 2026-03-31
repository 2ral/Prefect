from ...utils.ccov import extract_functions
from .grammar import BC_GRAMMAR

from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer, convert_and_validate_ebnf_grammar
from modelizer.configs import SEED

from typing import Optional
import subprocess
import signal
from os.path import normpath, dirname, abspath, join, exists
from os import listdir, remove, setsid, killpg, getpgid


# subject class
class BC_Subject(BaseSubject):
    def __init__(self, timeout: Optional[int] = 5, trials = None):
        super().__init__(timeout, trials)
        file_directory = normpath(dirname(abspath(__file__)))
        self._bc_directory = file_directory + '/compiled_bc-1.08.1/bc/'
        self._bc_command = "("\
                           f"cd {self._bc_directory};"\
                           'echo "{}" | ./bc -l;'\
                           ")"
        self._gcovr_command = "("\
                              f"cd {self._bc_directory};"\
                              "gcovr --json coverage.json;"\
                              ")"

    def __getstate__(self):
        state = super().__getstate__()
        state["bc_command"] = self._bc_command
        state["gcovr_command"] = self._gcovr_command
        state["bc_directory"] = self._bc_directory
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._bc_command = state["bc_command"]
        self._gcovr_command = state["gcovr_command"]
        self._bc_directory = state["bc_directory"]

    def __execute__(self, data: str | list | tuple) -> str | list:
        data = data.replace(" . ", ".")
        data = data.replace("  ", " ")
        self._input = data
        bc_args = self._bc_command
        bc_args = bc_args.format(self._input)
        gcovr_args = self._gcovr_command

        # run bc command
        try:
            bc_proc = subprocess.Popen(bc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, preexec_fn=setsid)
            stdout, stderr = bc_proc.communicate(timeout=self._timeout)
            # print(stderr)
        except subprocess.TimeoutExpired:
            self._state = ExecutionState.TIMEOUT
            # kill process
            killpg(getpgid(bc_proc.pid), signal.SIGTERM)
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
        else:
            rc = bc_proc.returncode if bc_proc else None
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
        if exists(self._bc_directory + "/coverage.json"):
            try:
                # extract coverage from json
                raw_coverage = extract_functions(self._bc_directory + "/coverage.json")
                coverage = []
                for file in raw_coverage:
                    coverage.extend(file.tokens())
                if self._state == ExecutionState.TIMEOUT:
                    self._output = [str(self._state.value), "ERROR:_BC_or_GCovr_Timeouted"] + coverage
                else:
                    self._output = [str(self._state.value)] + coverage
            except Exception as e:
                self._state = ExecutionState.EXCEPTION
                self._output = [str(self._state.value), "ERROR:_Unable_To_Read_Coverage_Report"]
        else:
            self._state = ExecutionState.EXCEPTION
            self._output = [str(self._state.value), "ERROR:_Coverage_Report_Not_Found"]

        # clean up and delete gcda files and coverage.json
        for filename in listdir(self._bc_directory):
            if filename.endswith(".gcda"):
                filepath = join(self._bc_directory, filename)
                remove(filepath)
            if filename == 'coverage.json':
                filepath = join(self._bc_directory, filename)
                remove(filepath)
        return self._output


# generator class
class BC_Generator(GeneratorInterface):
    def __init__(self, subject:BC_Subject, seed=SEED, logger=None, fuzzer_factory=None, min_nonterminals=1, max_nonterminals=24):
        super().__init__(source="bc", target="coverage", subject=subject, seed=seed, logger=logger)
        self._grammar = convert_and_validate_ebnf_grammar(BC_GRAMMAR)
        if fuzzer_factory == None:
            self._fuzzer = GrammarCoverageFuzzer(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)
        else:
            self._fuzzer = fuzzer_factory(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)

    def generate(self):
        data_str = self._fuzzer.fuzz()
        return data_str
