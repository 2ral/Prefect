from .grammar import DATEUTIL_GRAMMAR
from modelizer.dependencies.debuggingbook import FullCoverageTracer
from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer
from modelizer.configs import SEED
from func_timeout import func_timeout, FunctionTimedOut

# subject imports
import dateutil

# subject class
class Dateutil_Subject(BaseSubject):
    def __init__(self, timeout = 2, trials = None, quick_start = False):
        super().__init__(timeout, trials, quick_start)

    def __getstate__(self):
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

    def __execute__(self, data):
        self._input = data
        # init tracer
        tracer = FullCoverageTracer(obj=dateutil.parser.isoparse)
        # target function in trace window
        def trace():
            with tracer:
                dt = dateutil.parser.isoparse(self._input)
        try:
            # execute target in timeout window
            func_timeout(self.timeout, trace)
            # prepare state/output
            self._state = ExecutionState.PASS
            self._output = [str(self._state.value)]
        except FunctionTimedOut:
            self._state = ExecutionState.TIMEOUT
            self._output = [str(self._state.value), "ERROR:_Function_Timeouted"]
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
            self._output = [str(self._state.value), f"ERROR:_Exception_Occured:{e}"]
        # read out buffer and return
        executed_lines = tracer.executed_lines()
        # transform data
        for key,value in executed_lines.items():
            for line in value:
                self._output.append(f"{key} {line}")
        return self._output


# generator class
class Dateutil_Generator(GeneratorInterface):
    def __init__(self, subject:Dateutil_Subject, seed=SEED, logger=None, fuzzer_factory=None, min_nonterminals=5, max_nonterminals=20):
        super().__init__(source="dateutil", target="trace", subject=subject, seed=seed, logger=logger)
        self._grammar = DATEUTIL_GRAMMAR
        if fuzzer_factory == None:
            self._fuzzer = GrammarCoverageFuzzer(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)
        else:
            self._fuzzer = fuzzer_factory(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)

    def generate(self):
        fuzzed = self._fuzzer.fuzz()
        return fuzzed
