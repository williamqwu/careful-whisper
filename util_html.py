# util_html.py
import logging, math, re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from string import Template

try:
    import pysbd  # best-quality sentence boundary detection

    _PYSBD = True
except Exception:
    _PYSBD = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audio Transcription</title>
<style>
  :root {
    --fg: #111;
    --bg: #f6f6f7;
    --panel: #fff;
    --muted: #666;
    --chip: #f2f2f2;
  }
  body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; background: var(--bg); color: var(--fg); transition: background .3s, color .3s; }
  body.dark { --fg: #f1f1f1; --bg: #181818; --panel: #222; --muted: #aaa; }
  header { background: #333; color: #fff; padding: 1rem; font-size: 1.3rem; font-weight: bold; text-align: center; }
  .wrap { max-width: 880px; margin: 1rem auto; padding: 0 1rem; }
  .info { background: var(--panel); box-shadow: 0 2px 6px rgba(0,0,0,.08); padding: 1rem 1.25rem; border-radius: 10px; }
  .info ul { list-style: none; padding: 0; margin: 0; display: grid; gap: .35rem; }
  .info li { color: var(--muted); }
  .audio { margin-top: .75rem; }
  .controls { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }
  .controls button { background: #555; color: #fff; border: none; border-radius: 5px; padding: .4rem .8rem; cursor: pointer; font-size: .9rem; }
  .controls button:hover { background: #777; }
  .content { margin-top: 1rem; display: grid; gap: .9rem; }
  .content p { background: var(--panel); box-shadow: 0 2px 6px rgba(0,0,0,.06); padding: 1rem 1.25rem; border-radius: 10px; }
  .sen { padding: 0 .15rem; border-radius: 4px; transition: background .2s; }
  .no-colors .sen { background: none !important; }
  .legend { font-size: .9rem; color: var(--muted); display: flex; align-items: center; gap: .5rem; margin-top: .5rem; }
  .grad { height: 10px; flex: 1; border-radius: 999px; background: linear-gradient(90deg, rgb(255,0,0), rgb(255,255,0), rgb(0,255,0)); }
  .legend span { white-space: nowrap; }
  details summary { cursor: pointer; font-weight: bold; margin-bottom: .5rem; }
</style>
</head>
<body>
<header>Audio Transcription</header>
<div class="wrap">
  <section class="info">
    <details open>
      <summary>Transcript Info</summary>
      <ul>
        <li><strong>File:</strong> $title</li>
        <li><strong>Generated:</strong> $created_date</li>
        <li><strong>Model:</strong> $model_name</li>
        <li><strong>Elapsed:</strong> $elapsed_time</li>
      </ul>
    </details>
    $audio_block
    <div class="controls">
      <button onclick="toggleColors()">Toggle Confidence Colors (c)</button>
      <button onclick="toggleDark()">Toggle Dark Mode</button>
    </div>
    <div class="legend"><span>Low confidence</span><div class="grad"></div><span>High</span></div>
  </section>
  <section class="content" id="content">
    $content
  </section>
</div>
<script>
  function toggleColors() {
    document.getElementById('content').classList.toggle('no-colors');
  }
  function toggleDark() {
    document.body.classList.toggle('dark');
  }
  // keyboard shortcut: press "c" to toggle confidence colors
  document.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'c') toggleColors();
  });
</script>
</body>
</html>
"""


def probability_to_rgb(prob: float) -> Tuple[int, int, int]:
    """Map probability in [0,1] to red→yellow→green ramp."""
    if prob < 0.0 or prob > 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    if prob <= 0.5:
        r, g = 255, int(255 * (prob * 2))
    else:
        r, g = int(255 * (2 - prob * 2)), 255
    return (r, g, 0)


# --- sentence tokenization helpers ---


def _tokenize_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if _PYSBD:
        seg = pysbd.Segmenter(language="en", clean=True)
        return [s.strip() for s in seg.segment(text) if s.strip()]
    # Fallback: split on ., ?, !, … while trying to avoid common abbrev glitches
    abbrev = r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|Eq|e\.g|i\.e)"
    pieces, buf = [], []
    for tok in re.findall(r"\S+|\s+", text):
        buf.append(tok)
        if re.search(r"[.?!…][\"\')]?$", tok.strip()):
            # check last word isn't an abbreviation
            last = re.sub(r"[^A-Za-z\.\']+", "", tok)
            if re.fullmatch(rf"{abbrev}\.", last):
                continue
            pieces.append("".join(buf).strip())
            buf = []
    if buf:
        pieces.append("".join(buf).strip())
    return pieces


@dataclass
class Piece:
    text: str
    prob: float  # probability in [0,1]
    start_t: float  # segment start
    end_t: float  # segment end
    start_i: int = 0  # char offsets in concatenated string (filled later)
    end_i: int = 0


def _concat_pieces(segments) -> Tuple[str, List[Piece], List[int]]:
    """
    Build a single transcript string from whisper segments and keep:
    - per-piece char spans
    - paragraph-break offsets suggested by long pauses between segments
    """
    pieces: List[Piece] = []
    break_offsets: List[int] = []
    s = []
    cursor = 0
    prev_end = None

    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            prev_end = seg.get("end", prev_end)
            continue
        prob = math.exp(seg.get("avg_logprob", -1.0))  # convert log prob → prob
        p = Piece(
            text=txt,
            prob=max(0.0, min(1.0, prob)),
            start_t=seg.get("start", 0.0),
            end_t=seg.get("end", 0.0),
        )
        # long pause between segments → suggest a break
        if prev_end is not None:
            gap = max(0.0, p.start_t - prev_end)
            if gap >= 2.0:  # paragraph-level pause (tunable)
                break_offsets.append(cursor)  # at previous char position
        # add a single space between pieces
        if s:
            s.append(" ")
            cursor += 1
        p.start_i = cursor
        s.append(txt)
        cursor += len(txt)
        p.end_i = cursor
        pieces.append(p)
        prev_end = p.end_t
    return "".join(s), pieces, break_offsets


def _char_weighted_prob(span_start: int, span_end: int, pieces: List[Piece]) -> float:
    """Average probability over pieces overlapped by [span_start, span_end)."""
    total, wsum = 0.0, 0
    for p in pieces:
        a = max(span_start, p.start_i)
        b = min(span_end, p.end_i)
        if b > a:
            w = b - a
            total += p.prob * w
            wsum += w
        if p.start_i >= span_end:
            break
    return (total / wsum) if wsum else 0.5  # neutral if unknown


def _sentences_with_spans(text: str) -> List[Tuple[int, int, str]]:
    """Tokenize into sentences and recover their char spans via greedy search."""
    sents = _tokenize_sentences(text)
    spans = []
    pos = 0
    for s in sents:
        i = text.find(s, pos)
        if i == -1:
            # fallback: approximate using length
            i = pos
        j = i + len(s)
        spans.append((i, j, s))
        pos = j
    return spans


def _paragraphize(
    sent_spans: List[Tuple[int, int, str]],
    break_offsets: List[int],
    target_chars: int = 500,
) -> List[List[Tuple[int, int, str]]]:
    """Group sentences into paragraphs using (a) long-pause hints and (b) size budget."""
    breaks = set(break_offsets)
    paras, cur, cur_chars = [], [], 0
    # convert breaks to a sorted list for efficient checks
    breaks = sorted(breaks)

    def _has_break(upto_idx):
        # any break offset <= upto_idx
        return any(b <= upto_idx for b in breaks)

    last_break_index = -1
    for i, j, s in sent_spans:
        cur.append((i, j, s))
        cur_chars += j - i
        # break if we just crossed a long-pause boundary or exceeded size
        crossed_long_pause = False
        for b in breaks:
            if last_break_index < b <= j:
                crossed_long_pause = True
                last_break_index = b
                break
        if crossed_long_pause or cur_chars >= target_chars:
            paras.append(cur)
            cur, cur_chars = [], 0
    if cur:
        paras.append(cur)
    return paras


def create_highlighted_html(
    result,
    output_file,
    summary,
    sentence_per_par: int = 15,
    audio_src: Optional[str] = None,
):
    """
    Build readable HTML:
      - real sentence boundaries (pysbd if available)
      - paragraph breaks at long pauses and size budget
      - confidence color per sentence
    """
    logger = logging.getLogger()
    segments = result.get("segments", []) or []
    transcript, pieces, break_offsets = _concat_pieces(segments)
    sent_spans = _sentences_with_spans(transcript)
    paras = _paragraphize(sent_spans, break_offsets, target_chars=500)

    html_paragraphs = []
    for para in paras:
        chunks = []
        for i, j, s in para:
            prob = _char_weighted_prob(i, j, pieces)
            r, g, b = probability_to_rgb(prob)
            chunks.append(
                f'<span class="sen" style="background-color: rgb({r},{g},{b})">{s}</span>'
            )
        html_paragraphs.append("<p>" + " ".join(chunks) + "</p>")

    audio_block = ""
    if audio_src:
        # simple embedded audio; the sentences are not time-synced, but this is handy
        audio_block = (
            f'<div class="audio"><audio controls src="{audio_src}"></audio></div>'
        )

    file_data = dict(summary)
    file_data.setdefault("audio_block", "")  # ensure key exists
    file_data["content"] = "\n".join(html_paragraphs)

    html = Template(HTML_TEMPLATE).safe_substitute(file_data)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
        logger.info(f"Result HTML file created as {output_file}.")
