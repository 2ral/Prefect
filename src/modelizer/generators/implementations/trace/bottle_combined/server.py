# import requests
from bottle import Bottle, request, run, static_file, auth_basic, HTTPResponse
from os.path import normpath, dirname, abspath
from requests.auth import HTTPBasicAuth


html_dir = normpath(dirname(abspath(__file__))) + "/html_files"


global auth_flag_global
auth_flag_global = {
    'Nico': False,
    'Marius': False,
    'Tural': False
}
credentials = {
    "Nico": 'CovLearn',
    "Marius": 'LearnCov',
    "Tural": 'CISPA'
}


def run_server(port):
    app = Bottle()
    setup_routing(app)
    server = run(app, port=port, host='127.0.0.1', reloader=False)
    return server


def reset_server():
    global auth_flag_global
    auth_flag_global['Nico'] = False
    auth_flag_global['Marius'] = False
    auth_flag_global['Tural'] = False


def setup_routing(app: Bottle):
    app.route('/', ['GET'], index)
    app.route('/hello', ['GET'], hello)
    app.route('/random', ['GET'], random_number)
    app.route('/<filename:path>', ['GET'], callback=lambda filename: serve_static(filename))
    app.route('/<filename:path>', ['POST'], callback=lambda filename: post(filename))
    app.route('/<filename:path>', ['HEAD'], callback=lambda filename: head(filename))
    app.route('/<filename:path>', ['PUT'], callback=lambda filename: put(filename))
    app.route('/<filename:path>', ['DELETE'], callback=lambda filename: delete(filename))
    app.route('/<filename:path>', ['OPTIONS'], callback=lambda filename: options(filename))
    # auth user
    app.route('/protected.html', ['GET'], callback=protected)
    # auth server
    app.route('/hidden', ['GET'], callback=hidden)
    app.route('/login', ['GET'], callback = login)
    app.route('/logout', ['GET'], callback = logout)


def init_server(port):
    import requests
    requests.get(url=f"http://127.0.0.1:{port}/")           # index
    requests.get(url=f"http://127.0.0.1:{port}/hello")      # hello
    requests.get(url=f"http://127.0.0.1:{port}/random")     # random
    requests.get(url=f"http://127.0.0.1:{port}/abc")        # abc
    requests.post(url=f"http://127.0.0.1:{port}/abc")       # post
    requests.head(url=f"http://127.0.0.1:{port}/abc")       # head
    requests.put(url=f"http://127.0.0.1:{port}/abc")        # put
    requests.delete(url=f"http://127.0.0.1:{port}/abc")     # delete
    requests.options(url=f"http://127.0.0.1:{port}/abc")    # options
    requests.get(url=f"http://127.0.0.1:{port}/about.html")     # about
    # auth user
    requests.get(url=f"http://127.0.0.1:{port}/protected.html")    # protected
    # auth server
    requests.get(url=f"http://127.0.0.1:{port}/login", auth=HTTPBasicAuth('Nico', ''))          # login
    requests.get(url=f"http://127.0.0.1:{port}/logout", auth=HTTPBasicAuth('Nico', ''))         # logout
    requests.get(url=f"http://127.0.0.1:{port}/hidden", auth=HTTPBasicAuth('Nico', ''))      # protected


# --------------------------------- GET -----------------------------------------
def hello():
    return "Hello, World!"

def index():
    return static_file('index.html', root=html_dir)

def random_number():
    return "Random number: 1337"

def serve_static(filename):
    return static_file(filename, root=html_dir)


# ------------------------------Handler for POST request---------------------------------
def post(filename):
    data = request.forms.get('data')
    if data:
        return f"Received data: {data}"
    else:
        return "No data received"


# --------------------------------Handler for HEAD request----------------------------------
def head(filename):
    return "HEAD"


# --------------------------------Handler for PUT request-----------------------------------
def put(filename):
    return "PUT request received"


# --------------------------------Handler for DELETE request---------------------------------
def delete(filename):
    return "DELETE request received"


# --------------------------------Handler for OPTIONS request---------------------------------
def options(filename):
    return "OPTIONS request received"


# --------------------------------- Auth User -----------------------------------------
def check_password(username, password):
    # please never NEVER implement a password check like this
    check = False
    if username in credentials:
        stored_password = credentials[username]
        if password == stored_password:
            check = True
    else:
        stored_password = credentials.values()
        if password in stored_password:
            check = False
    return check

@auth_basic(check_password)
def protected():
    return static_file('protected.html', root=html_dir)


# --------------------------------- Auth Server -----------------------------------------
def check_name(username, password):
    global auth_flag_global
    return (username in auth_flag_global) and (auth_flag_global[username])

def login():
    name, _ = request.auth
    global auth_flag_global
    if name in auth_flag_global:
        auth_flag_global[name] = True
    else:
        HTTPResponse(f'User {name} unknown', status=403)

def logout():
    name, _ = request.auth
    global auth_flag_global
    if name in auth_flag_global:
        auth_flag_global[name] = False
    else:
        HTTPResponse(f'User {name} unknown', status=403)

@auth_basic(check_name)
def hidden():
    return static_file('protected.html', root=html_dir)
