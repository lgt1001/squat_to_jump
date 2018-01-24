# About
This project is using speech_recognition to control and start Wechat Jump game, and use opencv_python and face_recognition to capture squat depth and calculate press time for game.


# Requirements
Python 3.6
numpy (1.13.3)  [https://pypi.python.org/pypi/numpy/]
opencv-python (3.3.0.10)    [https://pypi.python.org/pypi/opencv-python/]
face-recognition (1.0.0)    [https://pypi.python.org/pypi/face_recognition/]
SpeechRecognition (3.8.1)   [https://pypi.python.org/pypi/SpeechRecognition/]

Adb tools   [https://developer.android.com/studio/releases/platform-tools.html#download]
Android Phone


# Run

	python squat_to_jump.py

if you are using CMU Sphinx (works offline), you can start with:

	python squat_to_jump.py --sphinx


# Play
1. Start squat_to_jump.py
2. Stand in front of camera and say "I am ready".
3. You can squat if screen shows "Start squatting...".
4. When Wechat Jump game is over, you can say "I am ready" or squat to start game again.


# Links in Github
opencv-python   [https://github.com/skvark/opencv-python]
face_recognition [https://github.com/ageitgey/face_recognition]
SpeechRecognition [https://github.com/Uberi/speech_recognition]