from .grammar import SQL_SELECT_GRAMMAR, SQL_MARKERS
from modelizer.dependencies.debuggingbook import FullCoverageTracer
from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.generators.postprocessor import PlaceholderProcessor
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer
from modelizer.configs import SEED
from func_timeout import func_timeout, FunctionTimedOut
from msticpy.data.sql_to_kql import sql_to_kql
import msticpy.data.sql_to_kql

# subject class
class SQL2KQL_Subject(BaseSubject):
    def __init__(self, timeout = 2, trials = None, quick_start = False):
        super().__init__(timeout, trials, quick_start=True)

    def __getstate__(self):
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

    def __execute__(self, data):
        self._input = data
        # init tracer
        tracer = FullCoverageTracer(obj=msticpy.data.sql_to_kql)
        # target function in trace window
        def trace():
            with tracer:
                kql_query = sql_to_kql(self._input)
                kql_query.replace("  not", " not")
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
    
    def pre_execution(self):
        kql = sql_to_kql("SELECT X FROM Y;")


# generator class
class SQL2KQL_Generator(GeneratorInterface):
    def __init__(self, subject:SQL2KQL_Subject, seed=SEED, logger=None, fuzzer_factory=None, min_nonterminals=5, max_nonterminals=20):
        super().__init__(source="sql2kql", target="trace", subject=subject, seed=seed, logger=logger)
        self._processor = PlaceholderProcessor(SQL_MARKERS, SQL_SELECT_GRAMMAR)
        self._grammar = SQL_SELECT_GRAMMAR
        if fuzzer_factory == None:
            self._fuzzer = GrammarCoverageFuzzer(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)
        else:
            self._fuzzer = fuzzer_factory(grammar=self._grammar, min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, seed=seed)


    def _update_ids(self, sql_query: str) -> str:
        def __find_id_value__(d, search_pos: int, inverse: bool = False, terminal_symbol: str = " ") -> tuple[str, int, int]:
            search_space = d[:search_pos] if inverse else d[search_pos:]
            table_pos = search_space.find("TABLE")
            id_start_pos = table_pos + 6
            id_end_pos = search_space.find(terminal_symbol, table_pos + 7)
            id_value = search_space[id_start_pos:id_end_pos]
            if not inverse:
                id_start_pos += search_pos
                id_end_pos += search_pos
            return id_value, id_start_pos, id_end_pos

        # Align table names in JOIN conditions
        join_pos = sql_query.find("JOIN")
        while join_pos != -1:
            from_id = __find_id_value__(sql_query, join_pos, inverse=True)
            target_id = __find_id_value__(sql_query, join_pos)
            join_left = __find_id_value__(sql_query, target_id[2], terminal_symbol=".", )
            join_right = __find_id_value__(sql_query, join_left[2], terminal_symbol=".", )
            # update left join
            sql_query = sql_query[:join_left[1]] + from_id[0] + sql_query[join_left[2]:]
            # update right join
            sql_query = sql_query[:join_right[1]] + target_id[0] + sql_query[join_right[2]:]
            # check for other joins
            join_pos = sql_query.find("JOIN", join_pos + 1)
        return sql_query


    def generate(self):
        sql = self._fuzzer.fuzz()
        sql = self._processor.deduplicate_placeholders(sql)
        sql = self._update_ids(sql)
        return sql
