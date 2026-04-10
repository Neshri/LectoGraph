import logging
from openscenesense_ollama.models import AnalysisPrompts
from lectograph.transcriber import FasterWhisperAdapter
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

transcriber = FasterWhisperAdapter(
    model_name="KBLab/kb-whisper-large",
    device="cuda"
)

custom_prompts = AnalysisPrompts(
    frame_analysis=(
        "Du analyserar en skärmbild från en IT-lektion. "
        "Beskriv endast vad som är direkt synligt: öppna fönster, programnamn, "
        "terminalutskrifter, fil- och mappstrukturer, dialogrutor och text på skärmen. "
        "Dra inga slutsatser eller antaganden om sådant som inte syns direkt."
    ),
    detailed_summary=(
        "Du sammanfattar en IT-lektion från en skärminspelning. "
        "Baserat strikt på skärmtidslinjen och ljudtranskriptionen nedan, "
        "skriv en trovärdig sammanfattning på svenska av vad som undervisades. "
        "Lägg inte till information som inte finns i källmaterialet. "
        "Om ett specifikt värde såsom en kod, ett kommando eller en identifierare "
        "inte finns explicit i tidslinjen eller transkriptionen, utelämna det helt.\n\n"
        "Videolängd: {duration:.1f} sekunder\n\n"
        "Skärmtidslinje:\n{timeline}\n\n"
        "Ljudtranskription:\n{transcript}"
    ),
    brief_summary=(
        "Baserat strikt på denna {duration:.1f} sekunder långa IT-lektion, "
        "skriv en kortfattad sammanfattning på svenska av vad som demonstrerades och förklarades. "
        "Ta endast med det som finns explicit i tidslinjen och transkriptionen. "
        "Om ett specifikt värde såsom en kod, ett kommando eller en identifierare "
        "inte finns explicit i tidslinjen eller transkriptionen, utelämna det helt.\n\n"
        "Skärmtidslinje:\n{timeline}\n\n"
        "Transkription:\n{transcript}"
    )
)

analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="glm-ocr",
    summary_model="gemma4:31b",
    min_frames=5,
    max_frames=45,
    frames_per_minute=3.0,
    frame_selector=DynamicFrameSelector(threshold=70.0),
    audio_transcriber=transcriber,
    prompts=custom_prompts,
    request_timeout=600.0,
    request_retries=1,
    log_level=logging.INFO
)

video_path = "input/cisco23.mp4"
try:
    results = analyzer.analyze_video(video_path)
except Exception as exc:
    logging.getLogger(__name__).error("analyze_video raised an exception: %s", exc, exc_info=True)
    raise

output_file = "summary_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Brief Summary:\n")
    f.write(results.get('brief_summary', '(missing)'))
    f.write("\n\nDetailed Summary:\n")
    f.write(results.get('summary', '(missing)'))

print(f"\nResults have been written to {output_file}")