import os
import tqdm
import numpy as np
import cv2

class VideoOutput:#'avc1', 'mp4v', 
    prev_im = None
    t_video = 0
    def __init__(self, path=None, fps=None, cc='mp4v', cc_fallback='avc1', fixed_fps=False, show=None):
        self.path = path
        self.cc = cc
        self.cc_fallback = cc_fallback
        self.fps = fps
        self.fixed_fps = fixed_fps
        self._show = not path if show is None else show
        self.active = self.path or self._show

    # init/cleanup

    def __enter__(self):
        self.prev_im = None
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        self.prev_im = None
        if self._show:
            cv2.destroyAllWindows()
    
    # allow them to work in async context managers too
    async def __aenter__(self): return self.__enter__()
    async def __aexit__(self, *a): return self.__exit__(*a)

    def output(self, im, timestamp: float|None=None):
        '''Output a video frame with an optional timestamp (epoch seconds).
        '''
        # convert to uint8
        if issubclass(im.dtype.type, np.floating):
            im = (255*im).astype('uint8')

        if self.path:  # write to file
            if self.fixed_fps and timestamp is not None:
                self.write_video_fixed_fps(im, timestamp)
            else:
                self.write_video(im)
        if self._show:  # no file, display to screen
            self.show_video(im)


    # manually call

    _w = None
    def write_video(self, im):
        if not self._w:
            # try a couple video encodings
            ccs = [self.cc, self.cc_fallback]
            for cc in ccs:
                os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
                # open video writer
                self._w = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*cc),
                    self.fps, im.shape[:2][::-1], True)
                if self._w.isOpened():
                    break
                print(f"{cc} didn't work trying next...")
            else:
                raise RuntimeError(f"Video writer did not open - none worked: {ccs}")
        self._w.write(im)

    def write_video_fixed_fps(self, im, timestamp: float):
        if self.prev_im is None:
            self.prev_im = im
            self.t_video = t

        while self.t_video < t:
            self.write_video(self.prev_im)
            self.t_video += 1./self.fps
        self.write_video(im)
        self.t_video += 1./self.fps
        self.prev_im = im

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


class VideoInput:
    def __init__(self, 
            src, fps=None, size=None, give_time=True, 
            start_frame=None, stop_frame=None, 
            bad_frames_count=True, 
            include_bad_frame=False):
        self.src = src
        self.dest_fps = fps
        self.size = size
        self.bad_frames_count = bad_frames_count
        self.include_bad_frame = include_bad_frame
        self.give_time = give_time
        self.start_frame = start_frame
        self.stop_frame = stop_frame

    def __enter__(self):
        self.cap = cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.src}")
        self.src_fps = src_fps = cap.get(cv2.CAP_PROP_FPS)
        self.dest_fps, self.every = fps_cvt(src_fps, self.dest_fps)

        size = self.size or (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame = np.zeros(tuple(size)+(3,)).astype('uint8')

        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.total = total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"{total/src_fps:.1f} second video. {total} frames @ {self.src_fps} fps,",
              f"reducing to {self.dest_fps} fps" if self.dest_fps else '')
        self.pbar = tqdm.tqdm(total=int(total))
        return self

    def __exit__(self, *a):
        self.cap.release()

    def read_all(self, limit=None):
        ims = []
        with self:
            for t, im in self:
                if limit and t > limit/self.dest_fps:
                    break
                ims.append(im)
        return np.stack(ims)

    def __iter__(self):
        i = self.start_frame or 0
        while not self.total or self.pbar.n < self.total:
            ret, im = self.cap.read()
            self.pbar.update()

            if self.bad_frames_count: i += 1

            if not ret:
                self.pbar.set_description(f"bad frame: {ret} {im}")
                if not self.include_bad_frame:
                    continue
                im = self.frame
            self.frame = im

            if not self.bad_frames_count: i += 1
            # i = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.stop_frame and i > self.stop_frame:
                break

            if i%self.every:
                continue
            if self.size:
                im = cv2.resize(im, self.size)

            t = i / self.src_fps
            self.pbar.set_description(f"t={t:.1f}s")
            yield t if self.give_time else i, im


class FrameInput:
    def __init__(self, src, src_fps, fps, file_pattern='frame_{:010d}', give_time=True, fallback_previous=True):
        import glob
        if os.path.isdir(src):
            files = glob.glob(os.path.join(src, "*"))
            ext = ".png" if len(files) == 0 else "." + os.path.basename(files[0]).split(".")[-1]
            src = os.path.join(src, file_pattern + ext)

        self.src = src
        self.src_fps = src_fps
        self.dest_fps, self.every = fps_cvt(src_fps, fps)

        self.give_time = give_time
        self.fallback_previous = fallback_previous

    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self):
        import cv2
        fs = os.listdir(os.path.dirname(self.src))
        i_max = fname2index(max(fs))
#        self.dest_fps, every = fps_cvt(self.src_fps, self.fps)
        print(f'{self.src}: fps {self.src_fps} to {self.dest_fps}. taking every {self.every} frames')

        im = None
        for i in tqdm.tqdm(range(0, i_max+1, self.every)):
            t = i / self.src_fps if self.give_time else i

            f = self.src.format(i)
            if not os.path.isfile(f):
                tqdm.tqdm.write(f'missing frame: {f}')
                if self.fallback_previous and im is not None:
                    yield t, im
                continue

            im = cv2.imread(f)
            yield t, im

def fname2index(f, sep='_'):
    '''path/frame_00003.jpg -> 3'''
    return int(os.path.splitext(os.path.basename(f))[0].split(sep)[-1])

def fps_cvt(src_fps, dest_fps):
    every = max(1, int(round(src_fps / (dest_fps or src_fps))))
    dest_fps = src_fps / every
    return dest_fps, every


class FrameOutput:
    def __init__(self, src, fname_pattern='frame_{:010.0f}.png'):
        self.src = os.path.join(src, fname_pattern)

    def __enter__(self): 
        os.makedirs(os.path.dirname(self.src), exist_ok=True)
        return self

    def __exit__(self, *a): pass
    def output(self, im, t):
        cv2.imwrite(self.src.format(t), im)
