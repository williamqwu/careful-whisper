from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from contextlib import contextmanager
import tempfile
from typing import Iterator, Optional, Iterable
import argparse
import textwrap

import ffmpeg
import whisper
import colorlog
from tqdm import tqdm

import util_html
import util_general

# supported models
MODEL_LIST = ["tiny", "base", "small", "medium", "large", "turbo"]
# supported language
try:
    # whisper.tokenizer exposes LANGUAGES (set of names) and TO_LANGUAGE_CODE (name->code)
    # also in https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
except Exception:
    LANGUAGES, TO_LANGUAGE_CODE = set(), {}

# ------------------------------ Logging ---------------------------------


def configure_logging(
    log_dir: Path = Path("artifacts/logs"),
    log_file: str = "whisper.log",
    level: int = logging.INFO,
) -> None:
    """
    Create a console + file logger with no duplicate handlers.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)

    # remove old handlers (avoid duplicates when re-running)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # file handler
    fh = logging.FileHandler(log_dir / log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # console handler (colored)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger.addHandler(fh)
    logger.addHandler(ch)


logger = logging.getLogger(__name__)

# ------------------------------ Helpers ---------------------------------


def _list_supported_models_text() -> str:
    return ", ".join(MODEL_LIST)


def _list_supported_languages_text() -> str:
    if not LANGUAGES:
        return "(auto-detect) or any BCP-47 code (e.g., en, fr, zh)"
    return ", ".join(sorted(LANGUAGES))


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def safe_stem(p: Path) -> str:
    # handles names like "foo.bar.mp4" --> "foo.bar"
    return p.name[: -(len(p.suffix))] if p.suffix else p.name


# -------------------------- Media Conversion ----------------------------


@contextmanager
def as_audio(
    path: Path,
    audio_format: str = "wav",
    keep_intermediate: bool = False,
    tmp_dir: Optional[Path] = None,
) -> Iterator[Path]:
    """
    Yield an audio file path for either an audio input (original file)
    or a video input (converted on-the-fly into a temp file).

    By default, temp audio is NOT persisted. Set keep_intermediate=True to keep it.
    """
    media_type = util_general.get_media_type(str(path))
    if media_type == "audio":
        yield path
        return
    if media_type != "video":
        raise TypeError(f"Unsupported media type for '{path}'")

    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # use NamedTemporaryFile but close it first (Windows-friendly)
    with tempfile.NamedTemporaryFile(
        prefix="whisper_", suffix=f".{audio_format}", dir=tmp_dir, delete=False
    ) as ntf:
        tmp_audio = Path(ntf.name)

    try:
        stream = ffmpeg.input(str(path))
        out = ffmpeg.output(
            stream, str(tmp_audio), format=audio_format, map="a", loglevel="quiet"
        )
        out = ffmpeg.overwrite_output(out)
        ffmpeg.run(out)
        logger.info(f"Converted video --> audio (temp): {tmp_audio}")
        yield tmp_audio
    finally:
        if tmp_audio.exists() and not keep_intermediate:
            try:
                tmp_audio.unlink()
                logger.debug(f"Deleted temp audio: {tmp_audio}")
            except Exception as e:
                logger.warning(f"Failed to delete temp audio {tmp_audio}: {e}")


# ----------------------------- Transcribe -------------------------------


def transcribe_audio(
    audio_file: Path,
    model_name: str,
    dump_json_to: Optional[Path] = None,
    word_timestamps: bool = False,
    language: Optional[str] = None,
) -> tuple[dict, float]:
    """
    Transcribe audio with Whisper and optionally dump the raw JSON.
    Returns (result_json, elapsed_seconds).
    """
    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_file),
        word_timestamps=word_timestamps,
        condition_on_previous_text=False,
        temperature=0.0,
        language=language,  # None = auto
    )
    dt = time.perf_counter() - t0
    logger.info(f"Transcription finished in {dt:.2f}s (model={model_name}).")

    if dump_json_to:
        dump_json_to.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_json_to, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote JSON --> {dump_json_to}")

    return result, dt


# --------------------------- Single Pipeline ----------------------------


def run_single_pipeline(
    input_file: Path,
    *,
    output_html_dir: Path = Path("artifacts/html"),
    output_json_dir: Optional[Path] = Path("artifacts/json"),
    logs_dir: Path = Path("artifacts/logs"),
    model: str = "base",
    audio_format: str = "wav",
    keep_intermediate_audio: bool = False,
    no_timestamp_to_name: bool = False,
    overwrite: bool = True,
    word_timestamps: bool = False,
    language: Optional[str] = None,
) -> Path:
    """
    Process one media file end-to-end:
      - produce audio (temp, not persisted by default)
      - transcribe with Whisper
      - render HTML (util_html)
      - optionally store raw JSON

    Returns the path to the generated HTML.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    output_html_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if output_json_dir:
        output_json_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_stem(input_file)
    # By default append "(YYMMDD_HHMM)" to the output filename; suppress with --no-timestamp-to-name
    if no_timestamp_to_name:
        out_name = f"{stem}.html"
    else:
        html_stamp = time.strftime("%y%m%d_%H%M")
        out_name = f"{stem} ({html_stamp}).html"
    html_path = output_html_dir / out_name
    if html_path.exists() and not overwrite:
        raise FileExistsError(f"{html_path} exists (overwrite=False).")

    json_path = (output_json_dir / f"{stem}.json") if output_json_dir else None

    with as_audio(
        input_file, audio_format=audio_format, keep_intermediate=keep_intermediate_audio
    ) as working_audio:
        result, elapsed = transcribe_audio(
            working_audio,
            model_name=model,
            dump_json_to=json_path,
            word_timestamps=word_timestamps,
            language=language,
        )

        summary = {
            "title": safe_stem(input_file),
            "model_name": model,
            "elapsed_time": f"{elapsed:.2f} sec",
            "created_date": now_tag(),
        }
        # pass audio_src so the player appears; it points at original audio (temp path works while file exists).
        util_html.create_highlighted_html(
            result, str(html_path), summary, audio_src=str(working_audio)
        )

    logger.info(f"HTML written --> {html_path}")
    return html_path


# ---------------------------- Batch Pipeline ----------------------------


def discover_media(
    input_dir: Path,
    exts: Iterable[str] = (".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv"),
    recursive: bool = True,
) -> list[Path]:
    """
    Return a sorted list of files with allowed extensions under input_dir.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    paths = []
    globber = input_dir.rglob if recursive else input_dir.glob
    for ext in exts:
        paths.extend(globber(f"*{ext}"))
    return sorted(set(paths))


def run_batch_pipeline(
    input_dir: Path,
    *,
    output_html_dir: Path = Path("artifacts/html"),
    output_json_dir: Optional[Path] = Path("artifacts/json"),
    logs_dir: Path = Path("artifacts/logs"),
    model: str = "base",
    audio_format: str = "wav",
    keep_intermediate_audio: bool = False,
    no_timestamp_to_name: bool = False,
    overwrite: bool = True,
    word_timestamps: bool = False,
    language: Optional[str] = None,
    recursive: bool = True,
) -> list[Path]:
    """
    Batch over a directory of media files. Returns list of generated HTML paths.
    """
    files = discover_media(Path(input_dir), recursive=recursive)
    if not files:
        logger.warning(f"No media files found in {input_dir}.")
        return []

    outputs: list[Path] = []
    for i, f in enumerate(tqdm(files, desc="Transcribing")):
        try:
            logger.info(f"[{i + 1}/{len(files)}] {f}")
            out = run_single_pipeline(
                f,
                output_html_dir=output_html_dir,
                output_json_dir=output_json_dir,
                logs_dir=logs_dir,
                model=model,
                audio_format=audio_format,
                keep_intermediate_audio=keep_intermediate_audio,
                no_timestamp_to_name=no_timestamp_to_name,
                overwrite=overwrite,
                word_timestamps=word_timestamps,
                language=language,
            )
            outputs.append(out)
        except Exception as e:
            logger.error(f"Failed on {f}: {e}")
    return outputs


# --------------------------------- Main ---------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    epilog = textwrap.dedent(f"""
    Supported models:
      {_list_supported_models_text()}.

    Supported languages:
      ==> Default: auto-detect
      ==> You may pass a language *name* (e.g., 'English', 'French') or a code (e.g., 'en', 'fr')
      ==> Known names (e.g., en): see https://github.com/openai/whisper/blob/main/whisper/tokenizer.py for details.
    """)
    p = argparse.ArgumentParser(
        description="Whisper-based transcription pipeline (single file or batch directory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog,
    )

    io = p.add_argument_group("I/O")
    io.add_argument(
        "path",
        type=Path,
        help="Path to a media file (single run) OR a directory (batch run).",
    )
    io.add_argument(
        "--output-html-dir",
        type=Path,
        default=Path("artifacts/html"),
        help="Directory to write highlighted HTML outputs.",
    )
    io.add_argument(
        "--output-json-dir",
        type=Path,
        default=Path("artifacts/json"),
        help="Directory to dump raw Whisper JSON (use --no-json to disable).",
    )
    io.add_argument(
        "--no-json",
        action="store_true",
        help="Disable writing raw JSON transcription outputs.",
    )
    io.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("artifacts/logs"),
        help="Directory for pipeline logs.",
    )

    run = p.add_argument_group("Run mode")
    run.add_argument(
        "--recursive",
        action="store_true",
        help="When PATH is a directory, search recursively for media files.",
    )
    run.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow overwriting existing HTML outputs.",
    )
    run.add_argument(
        "--no-timestamp-to-name",
        action="store_true",
        help="Disable appending (YYMMDD_HHMM) to output filename stem.",
    )

    model = p.add_argument_group("Model & transcription")
    model.add_argument(
        "--model",
        choices=MODEL_LIST,
        default="base",
        help="Which Whisper model to use.",
    )
    model.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force a language (name like 'English' or code like 'en'). Default: auto.",
    )
    model.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps (experimental).",
    )

    media = p.add_argument_group("Media conversion")
    media.add_argument(
        "--audio-format",
        type=str,
        default="wav",
        help="Temp audio format used when converting video to audio.",
    )
    media.add_argument(
        "--keep-intermediate-audio",
        action="store_true",
        help="Persist the converted temp audio file instead of deleting it.",
    )

    log = p.add_argument_group("Logging")
    log.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Console and file log level.",
    )
    log.add_argument(
        "--log-file",
        type=str,
        default="whisper.log",
        help="Log filename inside --logs-dir.",
    )

    return p


def _normalize_language_arg(lang: Optional[str]) -> Optional[str]:
    """Accept language name or code; map names to codes if possible."""
    if not lang:
        return None
    s = lang.strip()
    if not s:
        return None
    # Already a code?
    if len(s) <= 3 and s.lower() == s:
        return s
    # Try name -> code mapping if available
    try:
        return TO_LANGUAGE_CODE.get(s.lower(), s)
    except Exception:
        return s


if __name__ == "__main__":
    _ap = _build_arg_parser()
    args = _ap.parse_args()

    # logging
    configure_logging(
        log_dir=args.logs_dir,
        log_file=args.log_file,
        level=getattr(logging, args.log_level),
    )
    logger.info("Whisper pipeline starting")
    logger.info("=========================")

    output_json_dir: Optional[Path] = None if args.no_json else args.output_json_dir
    forced_lang = _normalize_language_arg(args.language)

    if args.path.is_dir():
        outs = run_batch_pipeline(
            args.path,
            output_html_dir=args.output_html_dir,
            output_json_dir=output_json_dir,
            logs_dir=args.logs_dir,
            model=args.model,
            audio_format=args.audio_format,
            keep_intermediate_audio=args.keep_intermediate_audio,
            no_timestamp_to_name=args.no_timestamp_to_name,
            overwrite=args.overwrite,
            word_timestamps=args.word_timestamps,
            language=forced_lang,
            recursive=args.recursive,
        )
        logger.info(f"Batch completed: {len(outs)} HTML file(s) written.")
    else:
        out = run_single_pipeline(
            args.path,
            output_html_dir=args.output_html_dir,
            output_json_dir=output_json_dir,
            logs_dir=args.logs_dir,
            model=args.model,
            audio_format=args.audio_format,
            keep_intermediate_audio=args.keep_intermediate_audio,
            no_timestamp_to_name=args.no_timestamp_to_name,
            overwrite=args.overwrite,
            word_timestamps=args.word_timestamps,
            language=forced_lang,
        )
        logger.info(f"Single file completed: {out}")
    logger.info("Pipeline ends successfully\n")
