from modelizer.generators.subjects import BaseSubject
from modelizer.utils import find_free_port


ENTRY_STR_BOTTLE= "socketserver.py serve_forever 237"
EXIT_STR_BOTTLE= "socketserver.py serve_forever 239"


def init_subject(subject_name: str) -> BaseSubject:
    subject_name = subject_name.lower()
    match subject_name:
        case "bc":
            from ..coverage.bc.bc import BC_Subject
            sbj = BC_Subject()
        case "grep":
            from ..coverage.grep.grep import Grep_Subject
            sbj = Grep_Subject()
        case "re2":
            from ..coverage.re2.re2 import Re2_Subject
            sbj = Re2_Subject()
        case "sql2kql":
            from ..trace.sql2kql.sql2kql import SQL2KQL_Subject
            sbj = SQL2KQL_Subject()
        case "dateutil":
            from ..trace.dateutils_subject.dateutils_subject import Dateutil_Subject
            sbj = Dateutil_Subject()
        case "bottle":
            from ..trace.bottle_combined.bottle_combined import Bottle_Subject
            free_port = find_free_port()
            sbj = Bottle_Subject(port=free_port)
        case _:
            raise ValueError(f"Invalid subject name: {subject_name}")
    return sbj
