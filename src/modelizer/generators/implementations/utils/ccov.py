import json
import os 
import fnmatch

class Line():
    def __init__(self, linenum, count) -> None:
        self.linenum = int(linenum)
        self.count = int(count)
    
    def is_executed(self):
        return self.count > 0
    
    def __lt__(self, other):
        return self.linenum < other.linenum
    
    def __eq__(self, other):
        return self.linenum == other.linenum

class Function():
    # NOTE executions is currently not stored correctly and is just holding dummy values !!!
    def __init__(self, name, begin, file, executions) -> None:
        self.name = name
        self.begin = begin
        self.end = 0
        self.file = file
        self.executions = executions
    
    def is_inside(self, num:int):
        return num >= int(self.begin) and num <= int(self.end)
    
    def __lt__(self, other):
        return self.begin < other.begin
    
    def __eq__(self, other):
        return self.begin == other.begin
    
    def __repr__(self):
        return f'Function {self.name}: line {self.begin} to {self.end} executed {self.executions}%\n'
    
class File():
    def __init__(self, name: str, functions: list[Function], linenumbers: list[Line]) -> None:
        self.name = name
        self.functions = functions
        self.linenumbers = linenumbers
    
    def __repr__(self) -> str:
        return f'\nFile {self.name} containing:\n\t{self.functions}\nand lines\n\t{[line.linenum for line in self.linenumbers]}'
    
    def tokens(self):
        functions = []
        lines = []
        for func in self.functions:
            functions.append(f"{self.name} {func.name} {func.begin} {func.end} {func.executions}")

        for line in self.linenumbers:
            #print(line)
            tmp_name = 'NONE'
            for func in self.functions:
                if func.is_inside(int(line.linenum)):
                    tmp_name = func.name
            if line.is_executed():
                lines.append(f"{self.name} {tmp_name} {line.linenum}")
        return lines
    

def find_end(functions:list[Function], line_numbers_:list[Line]):
    line_numbers = [line.linenum for line in line_numbers_]
    for i in range(len(functions)):
        if i == len(functions)-1:
            functions[i].end = max(line_numbers)
            break
        current = functions[i]
        next_func = functions[i+1]
        end_line_number = 0
        for line_number in line_numbers:
            if line_number < next_func.begin:
                end_line_number = line_number
            else:
                break
        current.end = end_line_number

def extract_functions(path:str):
    with open(path, 'r') as file:
        content = json.load(file)

    files = content['files']
    # print(len(files))
    coverage = [] # File(file, functions, linenumbers)

    for file in files:
        functions = []
        lines = []
        for function in file['functions']: # (name, beginn, ende)
            # NOTE second lineno entry is a dummy entry !!!!
            if 'demangled_name' in function.keys():
                functions.append(Function(function['demangled_name'], function['lineno'], file['file'], function['lineno']))
            else:
                functions.append(Function(function['name'], function['lineno'], file['file'], function['lineno']))

        for line in file['lines']: # line_number
            lines.append(Line(line['line_number'], line['count']))

        lines.sort()
        #print(lines)
        functions.sort()
        find_end(functions, lines)

        # for function in functions:
            # print(function)
        coverage.append(File(file['file'], functions, lines))
    
    return coverage



def find_functions_calls(directory, coverage):
    functions_list = []
    starting_lines = []
    for file in coverage:
        for function in file.functions:
            functions_list.append(function.name)
            starting_lines.append([function.name, function.begin])
    results = []

    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.c'):
            filepath = os.path.join(root, filename).replace('\\', '/')
            # print(f'looking at {filepath}')
            try:
                with open(filepath, 'r') as file:
                    content = file.read().split('\n')
                    # print(len(content))
                    for line in content:
                        for function in functions_list:
                            if function in line:
                                results.append((function, filepath, content.index(line)+1))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        for name, _, line in results:
            for entry in starting_lines:
                if entry[0] == name and entry[1] == line:
                    results.remove((name, _, line))

    return results



# coverage = extract_functions('slre/SLREdata/coverage.json')

# for file in coverage:
#     print(file.tokens())

# functions_calls = find_functions_calls(directory='cdata', coverage=coverage)
