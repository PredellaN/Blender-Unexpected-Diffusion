class ProcessManager:
    def __init__(self, ws, pg):
        self.ws = ws
        self.pg = pg

    def set_running(self, value):
        self.pg.running = value

    def set_progress(self, value):
        self.pg.progress = value

    def set_progress_text(self, value):
        self.pg.progress_text = value

    def set_stop_process(self, value):
        self.pg.stop_process = value

    def stop_process(self):
        return self.pg.stop_process

    def redraw(self):
        for screen in self.ws.screens:
            for area in screen.areas:
                area.tag_redraw()