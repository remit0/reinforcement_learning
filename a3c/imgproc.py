import cv2 as cv


class FrameTransformer:

    def normalize(self, frame):
        return frame * (1 / 255)

    def to_gray(self, frame):
        return cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    def resize(self, frame):
        return cv.resize(frame, dsize=(80, 80), interpolation=cv.INTER_NEAREST)

    def transform(self, frame):
        frame = self.to_gray(frame)
        frame = self.normalize(frame)
        frame = self.resize(frame)
        return frame
