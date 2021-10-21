import pyaudio
import wave
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

def record():

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)

	print("Listening...")

	frames = []
	t = time.time()
	try:
		while (time.time()-t < 3):
			data = stream.read(CHUNK)
			frames.append(data)
	except:
		print("Done recording")

	sample_width = p.get_sample_size(FORMAT)
	
	stream.stop_stream()
	stream.close()
	p.terminate()
	
	return sample_width, frames	

def record_to_file(file_path):
	wf = wave.open(file_path, 'wb')
	wf.setnchannels(CHANNELS)
	sample_width, frames = record()
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
