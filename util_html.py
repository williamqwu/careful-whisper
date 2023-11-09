import logging
import math

# A minimal html template for simplicity
HTML_TEMPLATE_MIN = """
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

# A more elegant html template
HTML_TEMPLATE = """
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

def probability_to_rgb(prob):
    """
    Convert log_prob to RGB color tuple with green yellow red color scale
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
    logger = logging.getLogger()
    
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
    
    file_data = summary
    file_data["content"] = " ".join(highlighted_sentences)
    html_content = HTML_TEMPLATE.format(**file_data)

    with open(output_file, "w") as f:
        f.write(html_content)
        logger.info(f"Result HTML file created as {output_file}.")
