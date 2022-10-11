"""
    This programs allow to create specialised agent for visually impaired people.

    Copyright (C) 2019 Vincent STRAGIER (vincent.stragier@outlook.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# from __future__ import print_function
# from functools import partial
# import argparse
import io
# import json
# import requests
# import socket
import locale
import threading
# import traceback
import os
import tempfile
# import pkg_resources
# import sys

from datetime import datetime
from contextlib import contextmanager

import pyttsx3
engine_tts = pyttsx3.init()

LOCALE_LOCK = threading.Lock()

@contextmanager
def setlocale(name):
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)

# Let's set a non-US locale
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

# KEYS_FILE = pkg_resources.resource_string(__name__, "evasion_keys.json")

# class ManageVerbose:
#     """Allows to mask print() to the user."""

#     def __init__(self, verbose=True):
#         self.verbosity = verbose

#     def __enter__(self):
#         if not self.verbosity:
#             self._original_stdout = sys.stdout
#             sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if not self.verbosity:
#             sys.stdout.close()
#             sys.stdout = self._original_stdout


def save_client_request(request: dict) -> bool:
    # TODO:
    # save request in an historic (SQL database)

    return False

# Dialogue manager


def handle_user_request(request: dict) -> dict:
    # Save request in database
    # save_client_request(request)
    # Resolve ellipse
    # request = resolve_ellipses(request)
    # Analyse request
    reply = ""
    return reply


def generate_answer(intent: str, request: str) -> str:
    match intent:
        case 'time':
            return 'Il est ' + datetime.today().strftime('%H heures %M minutes et %S secondes')
        case 'date':
            return 'Nous sommes le ' + datetime.today().strftime('%A %d %B %Y')
        case _:
            return f'"Réponse inconnue" ({intent = })'


def play(text: str, voice: int, rate: float, volume: float):
    voice = engine_tts.getProperty('voices')[voice]
    engine_tts.setProperty('voice', voice.id)
    engine_tts.setProperty('rate', rate)  # default is 200
    engine_tts.setProperty('volume', volume)
    engine_tts.say(text)
    engine_tts.runAndWait()


def generate_sound_file(text: str, voice: int, rate: float, volume: float):
    temp_file = io.BytesIO()
    with tempfile.TemporaryDirectory() as temp_dir:
        voice = engine_tts.getProperty('voices')[voice]
        engine_tts.setProperty('voice', voice.id)
        engine_tts.setProperty('rate', rate)  # default is 200
        engine_tts.setProperty('volume', volume)
        file_name = 'sound.mp3'
        file_name = os.path.join(temp_dir, file_name)
        engine_tts.save_to_file(text, file_name)
        engine_tts.runAndWait()
        while not os.listdir(temp_dir):
            pass
        temp_file = io.BytesIO(open(file_name, 'rb').read())
    return temp_file


def main() -> None:
    from flask import Flask, render_template, request, send_file
    from flask_socketio import SocketIO
    from . import chatbot

    app = Flask(__name__, template_folder="./")
    app.config['SECRET_KEY'] = 'aifvi!'
    socketio = SocketIO(app, cors_allowed_origins='*')

    # define app routes
    @app.route("/")
    def index():
        return render_template("index.htm")

    @app.route("/get")
    # function for the bot response
    def get_bot_response():
        # Do a simple echo
        userText = request.args.get("msg")
        # return f'echo: {userText}'
        predicted_intents = chatbot.predict_class(userText)
        # print(str(predicted_intents), "\n")
        # response = chatbot.get_response(predicted_intents, chatbot.intents)
        # print("Bot >>", response)
        intent = predicted_intents[0]["intent"]
        response = generate_answer(intent, userText)
        socketio.emit('speech', {'text': response, 'voice': 0, 'rate': 200, 'volume': 100})
        return f'Robot : {str(predicted_intents)}\n{response}'
        # return str(englishBot.get_response(userText))

    @app.route('/a')
    def generate_audio():
        text = request.args.get('text', '')
        voice = int(request.args.get('voice', '0'))
        rate = float(request.args.get('rate', '200'))
        volume = float(request.args.get('volume', '1'))
        # print(f'{text = }, {voice = }, {rate = }, {volume = }')
        return send_file(generate_sound_file(
            text=text, voice=voice, rate=rate, volume=volume),
            download_name='sound.mp3')

    # print(__name__)
    # if __name__ == "__main__":
    # app.run()
    socketio.run(app)


if __name__ == "__main__":
    import warnings
    mod = "interactive_agent"
    warnings.warn(
        f"use 'python -m {mod}', not 'python -m {mod}.{mod}'",
        DeprecationWarning)
    # main()
