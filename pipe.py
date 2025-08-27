# Env: conda activate whisper

import ffmpeg
import whisper
import os
import time

import json
from pathlib import Path
import logging
import colorlog
from tqdm import tqdm
import util_html
import util_general

MODEL_LIST = ['tiny','base','small','medium','large']

def configure_logging(log_file='whisper_pipe.log'):
    # Set up logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # prevent duplicate handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Create a stream handler to print log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_time_as_string():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr

def get_script_directory():
    """
    Get the directory of the script that runs this function.
    """
    try:
        script_path = os.path.abspath(__file__)
        return os.path.dirname(script_path)
    except NameError:
        # __file__ attribute may not work as expected when the code is run interactively, 
        # such as in a REPL or a Jupyter Notebook.
        logger.warning(f"__file__ attribute not working as expected. Using current working directory as an alternative. Script path: {script_path}.")
        return os.getcwd()
    
def video_to_audio(video_file, audio_file, audio_format='wav'):
    """
    Convert a video file to an audio file using FFmpeg.

    NOTE:
        Specify audio_format='mp3' to save disk usage with lower accuracy.
    """
    try:
        stream = ffmpeg.input(video_file)
        out = ffmpeg.output(stream, audio_file, format=audio_format, map='a', loglevel="quiet")
        out = ffmpeg.overwrite_output(out)
        ffmpeg.run(out)
        logger.info(f"Successfully converted {video_file} to {audio_file}")
    except ffmpeg.Error as e:
        logger.warning(f"Error during conversion: {e}")

def transcribe_audio(working_file, model_name, log_file, write_log=True, 
                     word_timestamps=False, language=None):
    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    result = model.transcribe(
        working_file,
        word_timestamps=word_timestamps,  # set True if you want finer pauses
        condition_on_previous_text=False,
        temperature=0.0,
        language=language,  # None = auto
    )
    dt = time.perf_counter() - t0
    logger.info(f"Transcribe finished. Elapsed time: {dt:.2f}s.")
    if write_log:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote JSON to {log_file}.")
    return result, dt

def run_single_pipeline(input_file, output_folder='output', cache_folder='cached_audios', 
                        audio_format='wav', model='base'):
    base = Path(get_script_directory())
    cache_folder_path = base / cache_folder
    output_folder_path = base / output_folder
    cache_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    input_type = util_general.get_media_type(input_file)
    stem = Path(input_file).name
    stem = Path(stem).stem  # remove one extension safely

    working_file = input_file
    if input_type == 'video':
        working_file = str(cache_folder_path / f"{stem}_audio.{audio_format}")
        video_to_audio(input_file, working_file, audio_format)
        logger.info(f"Converted video to audio: {working_file}")
    elif input_type == 'audio':
        logger.info("Audio detected as input.")
    else:
        raise TypeError("Input file type unknown.")

    raw_file_name = str(cache_folder_path / f"{stem}_raw.json")
    result, elapsed_time = transcribe_audio(working_file, model, raw_file_name, word_timestamps=False)

    output_file_name = str(output_folder_path / f"{stem}_output.html")
    summary = {
        "title": stem,
        "model_name": model,
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "created_date": get_time_as_string()
    }
    util_html.create_highlighted_html(result, output_file_name, summary, audio_src=working_file)
    
def run_batch_pipeline(input_folder, output_folder='output', cache_folder='cached_audios', audio_format='wav', model='base'):
    logger.debug('Entering batch mode')
    try:
        # Ensure the directory exists
        if not os.path.exists(input_folder):
            raise Exception(f"Directory '{input_folder}' does not exist.")

        # Ensure the given path is a directory
        if not os.path.isdir(input_folder):
            raise Exception(f"'{input_folder}' is not a directory.")

        # Iterate through the directory and list all files
        file_names = []
        for entry in os.listdir(input_folder):
            entry_path = os.path.join(input_folder, entry)
            if os.path.isfile(entry_path):
                file_names.append(entry_path)

    except Exception as e:
        logger.error(f"{e}")
        return

    for idx, file_name in enumerate(tqdm(file_names, desc='Processing')):
        try:
            logger.info(f"Testing file #{idx+1}/{len(file_names)} from batch: {file_name}")
            run_single_pipeline(file_name,output_folder=output_folder,cache_folder=cache_folder,audio_format=audio_format,model=model)
        except TypeError:
            continue

if __name__=='__main__':
    configure_logging()
    logger = logging.getLogger()
    logger.info('Whisper pipeline starting')
    logger.info('=========================')

    # run_batch_pipeline('batch_test',model='medium') # Enter your folder name here, or try `run_single_pipeline`
    run_single_pipeline('input_test/intro_29s.wav', model='medium', audio_format='mp3')

    logger.info('Pipeline ends successfully\n')
