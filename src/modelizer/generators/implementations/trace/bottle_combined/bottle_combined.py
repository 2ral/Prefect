# generation imports
from .server import run_server, init_server, reset_server
from .grammar import BOTTLE_GRAMMAR
from ...utils.tracer import CustomTracer
from ...utils.utils import ENTRY_STR_BOTTLE, EXIT_STR_BOTTLE

# modelizer imports
from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer, convert_and_validate_ebnf_grammar
from modelizer.configs import SEED

# other imports
from bottle import Bottle
import requests
from time import sleep
from multiprocessing import Process, Manager


# workaround to make tracing in subprocess possible
def traced_server(buffer, port):
    with CustomTracer(buffer, ENTRY_STR_BOTTLE, EXIT_STR_BOTTLE):
        run_server(port)


# subject class
class Bottle_Subject(BaseSubject):
    def __init__(self, port, timeout = None, trials = None, quick_start = True):
        self._port = port
        # initialiazing buffer
        manager = Manager()
        self.tracer_buffer = manager.list()
        self.server = Process(target=traced_server, args=(self.tracer_buffer, self._port))
        super().__init__(timeout, trials, quick_start)


    def __getstate__(self):
        state = super().__getstate__()
        state["port"] = self._port
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._port = state["port"]


    def pre_execution(self):
        # NOTE still utilizes old tracer setup
        # create server process to run in parallel
        self.server = Process(target=traced_server, args=(self.tracer_buffer, self._port))
        self.server.start()
        sleep(1) # sleep statement to make sure server finishes before any more requests can be send
        init_server(self._port)
        sleep(1)
        self.tracer_buffer[:] = []


    def post_execution(self):
        # terminate server and wrap up
        self.server.terminate()
        self.server.join(5)


    def reset(self):
        reset_server()


    def __execute__(self, data):
        reset_server()
        
        # replace fake port 80 with port that server actually runs on
        self._input:str = data
        self._input = self._input.replace("80", str(self._port))

        # clear buffer
        self.tracer_buffer[:] = []

        # send requests
        try:
            reqs = self._input.split("~")
            for req in reqs:
                req = req.split("#")
                method = req[1]
                actual_req = req[0]
                match method:
                    case 'GET':
                        status = requests.get(url=actual_req)
                    case 'HEAD':
                        status = requests.head(url=actual_req)
                    case 'OPTIONS':
                        status = requests.options(url=actual_req)
                    case 'POST':
                        status = requests.post(url=actual_req)
                    case 'PUT':
                        status = requests.put(url=actual_req)
                    case 'PATCH':
                        status = requests.patch(url=actual_req)
                    case 'DELETE':
                        status = requests.delete(url=actual_req)
            # prepare state/output
            self._state = ExecutionState.PASS
            self._output = [str(self._state.value)]
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
            self._output = [str(self._state.value), f"ERROR:_Exception_Occured:{e}"]
        
        # read out buffer
        executed_lines = list(self.tracer_buffer)
        # get coverage from trace
        self._output = list(dict.fromkeys(executed_lines))
        
        return self._output


# generator class
class Bottle_Generator(GeneratorInterface):
    def __init__(self, subject:Bottle_Subject, seed=SEED, logger=None, min_nonterminals=5, max_nonterminals=20):
        super().__init__(source="bottle", target="trace", subject=subject, seed=seed, logger=logger)
        self._grammar = convert_and_validate_ebnf_grammar(BOTTLE_GRAMMAR)
        self._fuzzer = GrammarCoverageFuzzer(grammar=self._grammar, seed=seed, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals)

    def generate(self):
        return self._fuzzer.fuzz()
