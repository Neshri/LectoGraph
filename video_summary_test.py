import argparse
import logging
from pathlib import Path

from lectograph.config import Config
from openscenesense_ollama.models import AnalysisPrompts
from lectograph.transcriber import FasterWhisperAdapter
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector

def main():
    parser = argparse.ArgumentParser(description="Test video summarization.")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        cfg = Config.from_yaml(Path(args.config))
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return

    transcriber = FasterWhisperAdapter(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        initial_prompt=cfg.whisper_initial_prompt
    )

    custom_prompts = AnalysisPrompts(
        frame_analysis=cfg.frame_analysis_prompt,
        detailed_summary=cfg.detailed_summary_prompt,
        brief_summary=cfg.brief_summary_prompt
    )

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=cfg.frame_analysis_model,
        summary_model=cfg.summary_model,
        min_frames=cfg.min_frames,
        max_frames=cfg.max_frames,
        frames_per_minute=cfg.frames_per_minute,
        frame_selector=DynamicFrameSelector(threshold=cfg.frame_threshold),
        audio_transcriber=transcriber,
        prompts=custom_prompts,
        request_timeout=cfg.request_timeout,
        request_retries=cfg.request_retries,
        log_level=logging.INFO
    )

    try:
        results = analyzer.analyze_video(args.video_path)
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

if __name__ == "__main__":
    main()