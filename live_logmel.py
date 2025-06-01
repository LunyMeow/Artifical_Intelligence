import numpy as np
import sounddevice as sd
from queue import Queue
import librosa
import time

class LiveLogMelExtractor:
    def __init__(self, sample_rate=16000, frame_length=0.025, frame_stride=0.01,
                 num_filters=40, fft_size=512, low_freq=0, high_freq=None):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.num_filters = num_filters
        self.fft_size = fft_size
        self.high_freq = high_freq or (sample_rate // 2)
        self.low_freq = low_freq

        self.frame_size = int(round(frame_length * sample_rate))
        self.frame_step = int(round(frame_stride * sample_rate))

        self.mel_filters = self._create_mel_filters()

        self.data_queue = Queue()
        self.is_listening = False
        self.output_stream = None


    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def _create_mel_filters(self):
        low_mel = self._hz_to_mel(self.low_freq)
        high_mel = self._hz_to_mel(self.high_freq)
        mel_points = np.linspace(low_mel, high_mel, self.num_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin = np.floor((self.fft_size + 1) * hz_points / self.sample_rate)

        filters = np.zeros((self.num_filters, int(self.fft_size // 2 + 1)))
        for i in range(1, self.num_filters + 1):
            left = int(bin[i - 1])
            center = int(bin[i])
            right = int(bin[i + 1])

            for k in range(left, center):
                filters[i - 1, k] = (k - bin[i - 1]) / (bin[i] - bin[i - 1])
            for k in range(center, right):
                filters[i - 1, k] = (bin[i + 1] - k) / (bin[i + 1] - bin[i])
        return filters

    def _calculate_logmel(self, signal):
        emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        signal_length = len(emphasized_signal)
        num_frames = 1 + (signal_length - self.frame_size) // self.frame_step

        frames = np.zeros((num_frames, self.frame_size))
        for i in range(num_frames):
            start = i * self.frame_step
            end = start + self.frame_size
            if end > signal_length:
                frames[i] = np.pad(emphasized_signal[start:], (0, end - signal_length), 'constant')
            else:
                frames[i] = emphasized_signal[start:end]

        frames *= np.hamming(self.frame_size)

        mag_frames = np.abs(np.fft.rfft(frames, n=self.fft_size))
        power_frames = (1.0 / self.fft_size) * (mag_frames ** 2)

        filter_banks = np.dot(power_frames, self.mel_filters.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        log_mel = 20 * np.log10(filter_banks)

        return log_mel

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Ses akışı hatası: {status}", flush=True)
        logmel_features = self._calculate_logmel(indata[:, 0])
        self.data_queue.put(logmel_features)

    def start(self):
        if self.is_listening:
            raise RuntimeError("Already listening...")

        self.is_listening = True
        print("Log-Mel extractor started. Listening... (Press Ctrl+C to stop)", flush=True)

        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                dtype='float32',
                samplerate=self.sample_rate,
                blocksize=self.frame_size
            ):
                while self.is_listening:
                    while not self.data_queue.empty():
                        logmel_data = self.data_queue.get()
                        audio = self.inverse_logmel_to_audio(logmel_data)
                        self.play_audio(audio)
                        yield logmel_data
                    time.sleep(0.01)  # Small sleep to prevent CPU overload

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            self.stop()
            raise e

    def stop(self):
        self.is_listening = False
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
        print("Listening stopped.", flush=True)

    def inverse_logmel_to_audio(self, logmel):
        # Convert log-mel to linear mel spectrogram
        mel_spec = librosa.db_to_power(logmel)
        
        # Convert to audio
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.fft_size,
            hop_length=self.frame_step,
            win_length=self.frame_size,
            window='hamming',
            power=1.0
        )
        
        # Normalize audio to prevent clipping
        if len(audio) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio

    def play_audio(self, audio):
        """Play the given audio through the speakers"""
        if len(audio) == 0:
            return
            
        # Stop any previous playback
        if self.output_stream is not None:
            self.output_stream.stop()
            
        # Play the audio in a non-blocking way
        self.output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        self.output_stream.start()
        self.output_stream.write(audio)
