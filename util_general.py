import ffmpeg
import logging


def get_media_type(file_path):
    logger = logging.getLogger()

    try:
        # Probe the file to get its metadata
        probe = ffmpeg.probe(file_path)

        # Check the streams to determine the media type
        has_audio, has_video = False, False
        for stream in probe["streams"]:
            if stream["codec_type"] == "audio":
                has_audio = True
            elif stream["codec_type"] == "video":
                has_video = True

        # Determine the media type based on the streams
        if has_audio and has_video:
            return "video"
        elif has_audio:
            return "audio"
        else:
            return "unknown"
    except ffmpeg.Error as e:
        logger.error(f"Error occurred while probing file: {e}")
        return "unknown"
