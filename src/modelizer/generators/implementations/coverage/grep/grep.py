# generation imports
from ...utils.ccov import extract_functions

# modelizer imports
from modelizer.generators.abstract import GeneratorInterface
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.configs import SEED

# other imports
from typing import Optional
from os.path import normpath, dirname, abspath, join, exists
from os import listdir, remove, setsid, killpg, getpgid
import signal
import subprocess
import random
import string


# subject class
class Grep_Subject(BaseSubject):
    def __init__(self, timeout: Optional[int] = None, trials = None):
        super().__init__(timeout, trials)
        file_directory = normpath(dirname(abspath(__file__)))
        self._grep_directory = file_directory + '/compiled_grep-3.11/src/'
        self._html_directory = file_directory + '/../../utils/Alice_Adventures_in_Wonderland_Project_Gutenberg.html'
        self._grep_command = "("\
                         f"cd {self._grep_directory};"\
                         f"cat {self._html_directory} | " + './grep "{}";'\
                         ")"
        self._gcovr_command = "("\
                              f"cd {self._grep_directory};"\
                              "gcovr --json coverage.json;"\
                              ")"

    def __getstate__(self):
        state = super().__getstate__()
        state["grep_command"] = self._grep_command
        state["gcovr_command"] = self._gcovr_command
        state["grep_directory"] = self._grep_directory
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._grep_command = state["grep_command"]
        self._gcovr_command = state["gcovr_command"]
        self._grep_directory = state["grep_directory"]

    def __execute__(self, data: str | list | tuple) -> str | list:
        self._input = data
        grep_args = self._grep_command
        grep_args = grep_args.format(self._input)
        gcovr_args = self._gcovr_command

        # run grep command
        try:
            grep_proc = subprocess.Popen(grep_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, preexec_fn=setsid)
            stdout, stderr = grep_proc.communicate(timeout=self._timeout)
            # print(stdout)
            # print(stderr)
        except subprocess.TimeoutExpired:
            self._state = ExecutionState.TIMEOUT
            # kill process
            killpg(getpgid(grep_proc.pid), signal.SIGTERM)
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
        else:
            rc = grep_proc.returncode if grep_proc else None
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
        if exists(self._grep_directory + "/coverage.json"):
            try:
                # extract coverage from json
                raw_coverage = extract_functions(self._grep_directory + "/coverage.json")
                coverage = []
                for file in raw_coverage:
                    coverage.extend(file.tokens())
                if self._state == ExecutionState.TIMEOUT:
                    self._output = [str(self._state.value), "ERROR:_Grep_or_GCovr_Timeouted"] + coverage
                else:
                    self._output = [str(self._state.value)] + coverage
            except Exception as e:
                self._state = ExecutionState.EXCEPTION
                self._output = [str(self._state.value), "ERROR:_Unable_To_Read_Coverage_Report"]
        else:
            self._state = ExecutionState.EXCEPTION
            self._output = [str(self._state.value), "ERROR:_Coverage_Report_Not_Found"]

        # clean up and delete gcda files and coverage.json
        for filename in listdir(self._grep_directory):
            if filename.endswith(".gcda"):
                filepath = join(self._grep_directory, filename)
                remove(filepath)
            if filename == 'coverage.json':
                filepath = join(self._grep_directory, filename)
                remove(filepath)
        return self._output


# generator class
class Grep_Generator(GeneratorInterface):
    def __init__(self, subject:Grep_Subject, seed=SEED, logger=None):
        super().__init__(source="grep", target="coverage", subject=subject, seed=seed, logger=logger)
        file_directory = normpath(dirname(abspath(__file__)))
        html_directory = file_directory + '/../../utils/Alice_Adventures_in_Wonderland_Project_Gutenberg.html'
        # create wordlist
        with open(html_directory, 'r') as file:
            content = file.read()
        self.wordlist = content.split()
        self.len_wordlist = len(self.wordlist)

    def _random_existing_sequence(self, n):
        start = random.randint(0, self.len_wordlist - n)
        return ' '.join(self.wordlist[start:start + n])
    
    def _random_mutation(self, n):
        chosen_words = random.choices(self.wordlist, k=n)
        mutated_words = []
        for word in chosen_words:
            if random.random() < 0.5 and len(word) > 1:
                split_word = list(word)
                action = random.choice(['replace', 'insert', 'delete'])
                idx = random.randint(0, len(split_word) - 1)
                ch = random.choice(''.join(string.ascii_letters + string.digits))
                if action == 'replace':
                    split_word[idx] = ch
                elif action == 'insert':
                    split_word.insert(idx, ch)
                elif action == 'delete':
                    del split_word[idx]
                word = ''.join(split_word)
            mutated_words.append(word)
        return ' '.join(mutated_words)

    def generate(self):
        # p = 1/3 odds for random mutation and 2/3 for existing sequence
        # n = randomly determine amount of words used for grep command
        p = random.randint(1, 3)
        n = random.randint(1, 3)
        if p==1:
            tokens = self._random_mutation(n)
        else:
            tokens = self._random_existing_sequence(n)
        return tokens
