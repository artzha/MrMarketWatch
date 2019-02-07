from flask import Flask
app = Flask(__name__)

urlpatterns = [
	
]

# run export FLASK_APP=index.py
# flask run
@app.route('/')
def index():
    return 'Index Page'

@app.route('/ai_version/version_id', methods=['POST'])
def analyze_data():
    return 'Hello, World!'