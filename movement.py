class Movement:
    """Allows the stepper motor to move"""
    def __init__(self, tilt, rotate, rpm):
        self.tilt = tilt
        self.rotate = rotate
        if rpm >= 0:
            self.rpm = rpm
        else:
            self.rpm = -rpm

    def move_left(self):

        return 't:00000r:' + rotate + 's:' + rpm

    def move_right(self):

        return 't:00000r:' + rotate + 's:' + rpm

    def move_up(self):

        return 't:' + tilt + 'r:00000s:' + rpm

    def move_down(self):

        return 't:' + tilt + 'r:00000s:' + rpm
