import requests

from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
bootstrap = Bootstrap(app)
socketio = SocketIO(app)

url = 'http://localhost:8000/ask'                                                                                               
                                                                                                                 

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    incoming_message = "You: " + message
    emit('message', incoming_message)

    response = requests.post(url, json={"text": message, "chat_id": session.get('chat_id', '')})
    print(response.json())
    outgoing_message = "Bot: " + response.json()['text']
    session['chat_id'] = response.json()['chat_id']
    emit('message', outgoing_message)

if __name__ == '__main__':
    socketio.run(app)
