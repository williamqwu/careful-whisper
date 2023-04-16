# Env: conda activate whisper

import ffmpeg
import os
import time
import whisper
import math

MODEL_LIST = ['tiny','base','small','medium','large']

def get_time_as_string():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr

def get_script_directory():
    """
    Get the directory of the script that runs this function.

    Returns:
        str: The directory of the script that runs this function.
    """
    try:
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
    except NameError:
        # __file__ attribute may not work as expected when the code is run interactively, 
        # such as in a REPL or a Jupyter Notebook.
        print(f"Warning: __file__ attribute not working as expected. Using current working directory as an alternative.")
        script_directory = os.getcwd()
    
    return script_directory

def get_media_type(file_path):
    try:
        # Probe the file to get its metadata
        probe = ffmpeg.probe(file_path)

        # Check the streams to determine the media type
        has_audio, has_video = False, False
        for stream in probe['streams']:
            if stream['codec_type'] == 'audio':
                has_audio = True
            elif stream['codec_type'] == 'video':
                has_video = True

        # Determine the media type based on the streams
        if has_audio and has_video:
            return 'video'
        elif has_audio:
            return 'audio'
        else:
            return 'unknown'
    except ffmpeg.Error as e:
        print(f"Error occurred while probing file: {e}")
        return 'unknown'

def video_to_audio(video_file, audio_file, audio_format='wav'):
    """
    Convert a video file to an audio file using FFmpeg.

    Args:
        video_file (str): The path to the input video file.
        audio_file (str): The path to the output audio file.
        audio_format (str): The format of the output audio file. Defaults to 'mp3'.

    Example usage:
        video_file = 'example_video.mp4'
        audio_file = 'example_audio.mp3'
        video_to_audio(video_file, audio_file)

    Note:
        Specify audio_format='mp3' to save disk usage with lower accuracy.
    """
    try:
        # Create an input stream from the video file
        input_stream = ffmpeg.input(video_file)

        # Create an output stream for the audio file with the specified format
        output_stream = ffmpeg.output(input_stream, audio_file, format=audio_format, map='a')

        # Run the FFmpeg command to perform the conversion
        ffmpeg.run(output_stream)
        print(f"Successfully converted {video_file} to {audio_file}")
    except ffmpeg.Error as e:
        print(f"Error occurred during conversion: {e}")

def transcribe_audio(working_file, model_name, log_file, write_log=True):
    t_start = time.process_time()
    model = whisper.load_model(model_name)
    result = model.transcribe(working_file)
    t_stop = time.process_time()
    print(f"Transcribe finished. Elasped time: {(t_stop-t_start)}.")
    if write_log:
        with open(log_file, 'w') as f:
            f.write(str(result))
        print(f"Result file created as {log_file}.")
    return result, (t_stop-t_start)

def probability_to_rgb(prob):
    """
    Convert log_prob to RGB color tuple with green yellow red color scale
    Example usage:
        probability = 0.3  # Change this value to test other probabilities
        rgb = probability_to_rgb(probability)
        print("RGB value:", rgb)

    NOTE: 
        in source code, the default logprob_threshold is -1.0 which means that any inference with prob. 
        lower than ~0.368 will be considered as failure.
        this feature is defined in line 399 of https://github.com/openai/whisper/blob/main/whisper/transcribe.py
    """
    if prob < 0.0 or prob > 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")

    if prob <= 0.5:
        r = 255
        g = int(255 * (prob * 2))
    else:
        r = int(255 * (2 - prob * 2))
        g = 255

    b = 0

    return (r, g, b)

def create_highlighted_html(result, output_file, summary, sentence_per_par=15):

    html_template_min = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Highlighted Sentences</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.5
            }}
            .highlighted {{
                display: inline;
                padding: 2px 5px;
                margin: 0;
            }}
        </style>
    </head>
    <body>
        <p>{}</p>
    </body>
    </html>
    """

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Transcription</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.5;
                margin: 0;
                padding: 0;
                background-color: #f0f0f0;
            }}

            header {{
                background-color: #333;
                color: #fff;
                padding: 1rem;
                font-size: 1.5rem;
                font-weight: bold;
                text-align: center;
            }}

            .info-container {{
                max-width: 800px;
                margin: 1rem auto;
                padding: 1rem;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}

            .info-container h1 {{
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }}

            .info-container ul {{
                list-style: none;
                padding: 0;
            }}

            .info-container ul li {{
                margin-bottom: 0.5rem;
                font-size: 1.1rem;
            }}

            .content-container p {{
                max-width: 800px;
                margin: 1rem auto;
                padding: 1rem;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <header>
            Audio Transcription
        </header>
        <section class="info-container">
            <ul>
                <li>File: {title}</li>
                <li>Generated Date: {created_date}</li>
                <li>Model: {model_name}</li>
                <li>Elapsed Time: {elapsed_time}</li>
            </ul>
        </section>
        <section class="content-container">
            <p>{content}</p>
        </section>
    </body>
    </html>
    """

    sentences = []
    rgb_colors = []

    for i in result['segments']:
        sentences.append(i['text'])
        rgb_colors.append(probability_to_rgb(math.exp(i['avg_logprob'])))

    if len(sentences) != len(rgb_colors):
        raise ValueError("The number of sentences must match the number of RGB colors.")

    highlighted_sentences = []
    cnt = 0
    for sentence, color in zip(sentences, rgb_colors):
        cnt += 1
        rgb_str = "rgb({}, {}, {})".format(*color)
        highlighted_sentence = '<span class="highlighted" style="background-color: {}">{}</span>'.format(rgb_str, sentence)
        highlighted_sentences.append(highlighted_sentence)
        if cnt==sentence_per_par:
            highlighted_sentences.append('<br><br>')
        cnt %= sentence_per_par

    # html_content = html_template.format(" ".join(highlighted_sentences))
    
    file_data = summary
    file_data["content"] = " ".join(highlighted_sentences)
    html_content = html_template.format(**file_data)

    with open(output_file, "w") as f:
        f.write(html_content)

def singlePipeline(input_file, cache_folder='cached_audios', audio_format='wav', model='base'):
    cache_folder_path = os.path.join(get_script_directory(), cache_folder)
    if not os.path.exists(cache_folder_path):
        os.makedirs(cache_folder_path)
        print(f"Warning: cached audio folder not existed. New cached folder created under {cache_folder_path}.")
    input_type = get_media_type(input_file)

    working_file = input_file
    if input_type == 'video':
        working_file = os.path.join(cache_folder_path, 'audio_'+get_time_as_string()+'.'+audio_format)
        video_to_audio(input_file, working_file, audio_format)
        print(f"Video detected as input. Corresponding audio file converted as {working_file}.")
    elif input_type == 'audio':
        print(f"Audio detected as input. No furthur action needed.")
    elif input_type == 'unknown':
        print(f"Input file type cannot be determined. Action aborted.")
        raise TypeError("Input file type unknown.")
    
    raw_file_name = os.path.join(cache_folder_path, 'audio_'+get_time_as_string()+'_raw.json')
    transcribe_result, elapsed_time = transcribe_audio(working_file, model, raw_file_name)
    
    output_file_name = os.path.join(cache_folder_path, 'audio_'+get_time_as_string()+'_output.html')
    summary = {
        "title": working_file,
        "model_name": model,
        "elapsed_time": "{0:.2f} sec".format(elapsed_time),
        "created_date": get_time_as_string()
    }
    create_highlighted_html(transcribe_result, output_file_name, summary)
    
# def batchPipeline()

if __name__=='__main__':
    singlePipeline('cached_audios/audio_20230416_154526.wav',model='medium')
