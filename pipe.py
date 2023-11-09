# Env: conda activate whisper

import ffmpeg
import whisper
import os
import time

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
        script_directory = os.path.dirname(script_path)
    except NameError:
        # __file__ attribute may not work as expected when the code is run interactively, 
        # such as in a REPL or a Jupyter Notebook.
        logger.warning(f"__file__ attribute not working as expected. Using current working directory as an alternative. Script path: {script_path}.")
        script_directory = os.getcwd()
    
    return script_directory

def video_to_audio(video_file, audio_file, audio_format='wav'):
    """
    Convert a video file to an audio file using FFmpeg.

    NOTE:
        Specify audio_format='mp3' to save disk usage with lower accuracy.
    """
    try:
        # Create an input stream from the video file
        input_stream = ffmpeg.input(video_file)

        # Create an output stream for the audio file with the specified format
        output_stream = ffmpeg.output(input_stream, audio_file, format=audio_format, map='a', loglevel="quiet")

        # Run the FFmpeg command to perform the conversion
        ffmpeg.run(output_stream)
        logger.info(f"Successfully converted {video_file} to {audio_file}")
    except ffmpeg.Error as e:
        logger.warning(f"Error occurred during conversion: {e}")

def transcribe_audio(working_file, model_name, log_file, write_log=True):
    t_start = time.process_time()
    model = whisper.load_model(model_name)
    result = model.transcribe(working_file)
    t_stop = time.process_time()
    logger.info(f"Transcribe finished. Elasped time: {(t_stop-t_start)}.")
    if write_log:
        with open(log_file, 'w') as f:
            f.write(str(result))
        logger.info(f"Result file created as {log_file}.")
    return result, (t_stop-t_start)

def run_single_pipeline(input_file, output_folder='output', cache_folder='cached_audios', audio_format='wav', model='base'):
    cache_folder_path = os.path.join(get_script_directory(), cache_folder)
    output_folder_path = os.path.join(get_script_directory(), output_folder)
    if not os.path.exists(cache_folder_path):
        os.makedirs(cache_folder_path)
        logger.warning(f"Cached audio folder not existed. New cached folder created under {cache_folder_path}.")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        logger.warning(f"Output folder not existed. New cached folder created under {output_folder_path}.")
    input_type = util_general.get_media_type(input_file)
    input_file_name = os.path.basename(input_file).replace('.', '_')

    working_file = input_file
    if input_type == 'video':
        working_file = os.path.join(cache_folder_path, input_file_name+'_audio.'+audio_format)
        video_to_audio(input_file, working_file, audio_format)
        logger.info(f"Video detected as input. Corresponding audio file converted as {working_file}.")
    elif input_type == 'audio':
        logger.info(f"Audio detected as input. No furthur action needed.")
    elif input_type == 'unknown':
        logger.warning(f"Input file type cannot be determined. File skipped.")
        raise TypeError("Input file type unknown.")
    
    raw_file_name = os.path.join(cache_folder_path, input_file_name+'_raw.json')
    transcribe_result, elapsed_time = transcribe_audio(working_file, model, raw_file_name)
    
    output_file_name = os.path.join(output_folder_path, input_file_name+'_output.html')
    summary = {
        "title": input_file_name,
        "model_name": model,
        "elapsed_time": "{0:.2f} sec".format(elapsed_time),
        "created_date": get_time_as_string()
    }
    util_html.create_highlighted_html(transcribe_result, output_file_name, summary)
    
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

    run_batch_pipeline('batch_test',model='tiny') # Enter your folder name here, or try `run_single_pipeline`

    logger.info('Pipeline ends successfully\n')
