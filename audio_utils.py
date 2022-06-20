import numpy as np
import librosa

import librosa.display
import matplotlib.pyplot as plt
import PIL.Image as pimg
from pydub import AudioSegment


def plot_mp3_matplot(filename):
    """
    plot_mp3_matplot -- using matplotlib to simply plot time vs amplitude waveplot
    
    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    
    Returns -- None
    """
    
    # sr is for 'sampling rate'
    # Feel free to adjust it
    x, sr = librosa.load(filename, sr=44100)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)


def convert_audio_to_spectogram(filename, sampling_rate=44100):
    """
    convert_audio_to_spectogram -- using librosa to simply plot a spectogram
    
    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    
    Returns -- None
    """
    
    # sr == sampling rate 
    x, sr = librosa.load(filename, sr=sampling_rate)
    
    # stft is short time fourier transform
    X = librosa.stft(x)
    
    # convert the slices to amplitude
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # ... and plot, magic!
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()


# same as above, just changed the y_axis from hz to log in the display func    
def convert_audio_to_spectogram_log(filename):
    x, sr = librosa.load(filename, sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'log')
    plt.colorbar()


def get_spectrogram(filename: str, sampling_rate: int = 44100):
    x, _ = librosa.load(filename, sr=sampling_rate)
    x_freq = librosa.stft(x)
    xdb = librosa.amplitude_to_db(abs(x_freq))
    return xdb


def get_spectrogram2(x: np.ndarray):
    x_freq = librosa.stft(x)
    xdb = librosa.amplitude_to_db(abs(x_freq))
    return xdb

def get_spectrogram3(x: np.ndarray):
    pass


def pad_repeat(image: pimg.Image, width: int)->pimg.Image:
    if image.width >= width:
        return image

    new_im = pimg.new('RGB', (width, image.height))
    offset = (width - image.width) // 2 % image.width

    if offset > 0:  # first part
        box = (image.width - offset, 0, image.width, image.height)
        new_im.paste(image.crop(box))

    while offset < width:
        new_im.paste(image, (offset, 0))
        offset += image.width
    return new_im


def has_element(np_array: np.ndarray) -> bool:
    return bool(np_array.ndim and np_array.size)


def get_min_from_array(np_array: np.ndarray) -> float:
    return np_array.flatten().min()


def get_max_from_array(np_array: np.ndarray) -> float:
    return np_array.flatten().max()


def is_even(num: int) -> bool:
    return num % 2 == 0


def pad_spectrogram(np_array: np.ndarray, target_size_w: int = 0, target_size_h: int = 0,
                    pad_value: int = -1) -> np.ndarray:
    # for safety
    if not has_element(np_array):
        return np.ndarray(list())

    # flags
    do_horizontal_padding = False
    do_vertical_padding = False
    pad_size_left = pad_size_right = pad_size_top = 0

    # obtain & calculate information
    arr_height, arr_width = np_array.shape

    # calculate padding size
    if arr_width < target_size_w:
        do_horizontal_padding = True
        pad_size_left = int((target_size_w - arr_width) / 2)
        pad_size_right = int((target_size_w - arr_width) / 2) \
            if is_even(target_size_w - arr_width) \
            else int((target_size_w - arr_width) / 2) + 1

    if arr_height < target_size_h:
        do_vertical_padding = True
        pad_size_top = target_size_h - arr_height

    # Confirm necessity to execute padding
    if not (do_horizontal_padding or do_vertical_padding):
        return np_array

    # define value to pad a spectrogram
    pad_value = pad_value if pad_value >= 0 else get_min_from_array(np_array)

    # Pad the array horizontally here
    np_array_pad = np.pad(np_array, ((pad_size_top, 0), (pad_size_left, pad_size_right)),
                          'constant', constant_values=pad_value)

    # Pad the array vertically here
    return np_array_pad


def mash_up_audio(filename: str, general_sound_file: str) -> np.ndarray:
    """
    References:
    https://stackoverflow.com/questions/40651891/
    https://stackoverflow.com/questions/31399903/
    https://stackoverflow.com/questions/38015319/

    :param filename:
    :param general_sound_file:
    :return:
    """
    audio2classify = AudioSegment.from_wav(filename)
    general_sound = AudioSegment.from_wav(general_sound_file)

    assert audio2classify.frame_rate == general_sound.frame_rate

    # concatenate background sounds.
    combined_bgs = AudioSegment.empty()
    if general_sound.duration_seconds < 1:
        while combined_bgs.duration_seconds < 1:
            combined_bgs += general_sound
    else:
        combined_bgs += general_sound

    # Mash up audios
    mixed_audio = audio2classify.overlay(combined_bgs, position=0)
    mixed_array = mixed_audio.get_array_of_samples()

    return np.array(mixed_array)[0:audio2classify.frame_rate]
