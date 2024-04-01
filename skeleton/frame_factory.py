from skeleton.g_adjust_frame import g_adjust_frame

class FrameFactory:
    frames = {
        'vanilla': g_adjust_frame,
        # TODO: future frame implementation
    }

    @staticmethod
    def get_frame(config):
        frame_class = FrameFactory.frames.get(config['model'])
        if frame_class:
            return frame_class(config)
        else:
            raise ValueError(f"Unknown model name: {config['model']}")