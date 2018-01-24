"""Use OpenCV and face_recognition to capture squat depth, and send press time to Android phone.
Also used speech_recognition to start Wechat Jump game."""
import argparse
import os
import threading
import time

import cv2
import face_recognition as fr
import numpy
import speech_recognition as sr

__author__ = 'Guangtu Liu'
__email__ = 'lgt.1001-@163.com'
__version__ = '0.1.1'


def search_image(target, source, threshold=0.8):
    """Search target from source image, and return its location.

    :param numpy.ndarray target: Target image. It must be not greater than the source image and have the same data type.
    :param numpy.ndarray source: Image where the search is running. It must be 8-bit or 32-bit floating-point.
    :param float threshold: threshold.
    :return tuple: Start location(x. y) of matched target.
    """
    res = cv2.matchTemplate(source, target, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val > threshold:
        return max_loc


class AndroidHelper(object):
    """Helper to control Wechat Jump game."""
    MY_SCREEN_WIDTH = 1440
    WECHAT_JUMP_MAX_DISTANCE = 455
    WECHAT_JUMP_MIN_DISTANCE = 125
    SENSITIVITY = 2.045

    def __init__(self):
        self.position = None
        self.start_image = cv2.imread('images/start.png')
        self.start_again_image = cv2.imread('images/start_again.png')
        self.resized = False

    def resize(self, width):
        """Resize image if screenshot's width is less than MY_SCREEN_WIDTH.

        :param int width: width of screenshot.
        """
        if not self.resized:
            if width < self.MY_SCREEN_WIDTH:
                def rz(image):
                    h, w, _ = image.shape
                    scale = width / self.MY_SCREEN_WIDTH
                    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

                self.start_image = rz(self.start_image)
                self.start_again_image = rz(self.start_again_image)
            self.resized = True

    def start_game(self):
        """Start Wechat Jump game."""
        if self.position is None:
            self.pull_image()
            if os.path.exists('screen.png'):
                state = cv2.imread('screen.png')
                self.resize(state.shape[1])
                pos = search_image(self.start_image, state)
                if pos is not None:
                    h, w, _ = self.start_image.shape
                    self.position = (pos[0] + w // 2, pos[1] + h // 2)
                # os.remove('screen.png')
        if self.position:
            cmd = 'adb shell input tap {} {}'.format(self.position[0], self.position[1])
            os.system(cmd)

    @classmethod
    def pull_image(cls):
        """Pull screenshot from phone to current directory."""
        os.system('adb shell screencap -p sdcard/screen.png')
        os.system('adb pull sdcard/screen.png screen.png')

    def is_game_over(self):
        """Check whether Wechat Jump game is over.

        :return bool: True if Wechat Jump game is over, otherwise False.
        """
        self.pull_image()
        if os.path.exists('screen.png'):
            pos = search_image(self.start_again_image, cv2.imread('screen.png'))
            if pos is not None:
                h, w, _ = self.start_again_image.shape
                self.position = (pos[0] + w // 2, pos[1] + h // 2)
                return True
            # os.remove('screen.png')
        return False

    def jump(self, press_time):
        """Press screen.

        :param int press_time: how long should be pressed.
        """
        pos = self.position if self.position is not None else (300, 300)
        cmd = 'adb shell input swipe {} {} {} {} {}'.format(pos[0], pos[1], pos[0], pos[1], press_time)
        os.system(cmd)


class MotionCapturer(object):
    """Capture video and get face location."""
    FACE_DISTANCE_GAP = 10

    def __init__(self, width, height):
        """Init.

        :param int width: Video width.
        :param int height: Video height.
        """
        self.video_width = width
        self.video_height = height

        self.init_location = None
        self.depth = None
        self.max_distance = None

    def alter(self, image, message):
        """Add message on the top of image.

        :param numpy.ndarray image: Image read from video.
        :param str message: message.
        """
        cv2.rectangle(image, (0, 0), (self.video_width, 36), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, message, (6, 24), font, 0.8, (255, 255, 255), 1)

    @classmethod
    def get_face_location(cls, image):
        """Get face location.

        :param numpy.ndarray image: Image read from video.
        :return: face location if it is found, otherwise None.
        """
        face_locations = fr.api.face_detector(image, 1)
        return face_locations[0] if face_locations else None

    def set_init(self, image):
        """Set init-location.

        :param numpy.ndarray image: Image read from video.
        :return bool: True if face is found and init-location is set, otherwise False.
        """
        print('Setting init location...')
        face_location = self.get_face_location(image)
        if face_location:
            self.init_location = face_location.center().y
            self.max_distance = self.video_height - face_location.height() / 2 - self.init_location - self.FACE_DISTANCE_GAP
            print('Init Face.y', self.init_location, 'Max Distance', self.max_distance)
            return True
        return False

    def capture_depth(self, image):
        """Capture squat depth.

        :param numpy.ndarray image: Image read from video.
        :return bool: True if depth is captured, otherwise False.
        """
        face_location = self.get_face_location(image)
        if face_location:
            current_face_y = face_location.center().y
            distance = current_face_y - self.init_location
            # print('Current Face.y', current_face_y, 'Current Distance', distance, 'Current Max Distance', self._current_distance)
            if self.depth is None or distance > self.depth:
                self.depth = distance
            elif self.depth is not None and self.FACE_DISTANCE_GAP < distance < self.depth:
                print('Found distance: %s' % self.depth)
                return True
        return False


class WechatJump(object):
    """Capture squat depth, voice and control Wechat Jump game."""
    NOT_READY = 0
    READY = 1
    IS_SET = 2

    INTERVAL = 2.0

    def __init__(self, video, use_sphinx=False):
        """Init.

        :param cv2.VideoCapture video:  VideoCapture object.
        """
        self.video = video
        self.voice = sr.Recognizer()
        self.game = AndroidHelper()
        self.motion = MotionCapturer(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self._status = self.NOT_READY
        self._is_jumping = False
        self._status_lock = threading.Lock()
        self._jumping_lock = threading.Lock()

        self._process = True
        self._relation = None
        self._capture_started = False

        self.recognize_method = self.voice.recognize_sphinx if use_sphinx else self.voice.recognize_google
        
    @property
    def status(self):
        """Get status. NOT_READY, READY or IS_SET.
        
        :return: status. NOT_READY, READY or IS_SET."""
        with self._status_lock:
            return self._status

    @status.setter
    def status(self, value):
        """Set status.
        
        :param value: NOT_READY, READY or IS_SET.
        """
        with self._status_lock:
            self._status = value

    @property
    def is_jumping(self):
        """Whether it is starting jumping for Wechat Jump game.
        
        :return bool: True if it is jumping, otherwise False."""
        with self._jumping_lock:
            return self._is_jumping

    @is_jumping.setter
    def is_jumping(self, value):
        """Set value for whether it is starting jumping for Wechat Jump game."""
        with self._jumping_lock:
            if not value:
                self.motion.depth = None
            self._is_jumping = value

    def _capture(self):
        """Capture voice and set status=READY."""
        while True:
            with sr.Microphone() as source:
                self.voice.adjust_for_ambient_noise(source, duration=2)
                audio = self.voice.listen(source, phrase_time_limit=5)
            try:
                print('Analyzing...')
                msg = self.recognize_method(audio)
                print('Analyzed:', msg)
                if 'ready' in msg:
                    print('Set ready')
                    self.status = self.READY
                    return
            except sr.UnknownValueError:
                print('Could not understand your voice')
            except sr.RequestError as e:
                print('Error; %s' % e)
                raise e
            time.sleep(0.1)

    def _jump(self):
        """Convert squat depth to press time, and send to Wechat Jump game."""
        press_time = int(numpy.rint(self.motion.depth * self._relation))
        print('Current Distance', self.motion.depth, 'Press Time', press_time)

        self.game.jump(press_time)
        time.sleep(self.INTERVAL + press_time/1000)
        if self.game.is_game_over():
            self.status = self.NOT_READY
            self._capture_started = False

        self.is_jumping = False

    def run(self):
        """Run process.
        1. if status is NOT_READY, it will start voice capturer.
        2. if status is READY, it will set the init-location.
        3. if status is IS_SET, it will start motion capturer and capture squat depth.
        """
        _, image = self.video.read()
        if self.status == self.NOT_READY:
            self.motion.alter(image, 'Please say "I am ready" to start game.')
            if not self._capture_started:
                self._capture_started = True
                threading.Thread(target=self._capture).start()
        elif self.status == self.READY and self.motion.set_init(image):
            self._relation = self.game.WECHAT_JUMP_MAX_DISTANCE / self.motion.max_distance * self.game.SENSITIVITY
            self.game.start_game()
            self.status = self.IS_SET
        elif self.status == self.IS_SET and not self.is_jumping:
            self.motion.alter(image, 'Start squatting...')
            if self._process:
                if self.motion.capture_depth(image):
                    self.is_jumping = True
                    threading.Thread(target=self._jump).start()
            self._process = not self._process

        cv2.imshow('Video', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sphinx', action='store_true', help='Use sphinx to recognize your voice.', default=False)
    args = parser.parse_args()
    video_capture = cv2.VideoCapture(0)
    my_jump = WechatJump(video_capture, args.sphinx)
    try:
        while True:
            my_jump.run()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
