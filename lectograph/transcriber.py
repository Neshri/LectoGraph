"""
Adapter for using faster-whisper with openscenesense_ollama.
"""

from __future__ import annotations

import logging
from typing import List

from faster_whisper import WhisperModel
from openscenesense_ollama.transcriber import AudioTranscriber
from openscenesense_ollama.models import AudioSegment


class FasterWhisperAdapter(AudioTranscriber):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Loading faster-whisper model '{model_name}' on {device}")
        compute_type = "float16" if device == "cuda" else "int8"
        
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, video_path: str) -> List[AudioSegment]:
        self.logger.info(f"Starting faster-whisper transcription for {video_path}")
        
        # condition_on_previous_text=False prevents the infinite looping/deadlock bug
        segments_generator, _ = self.model.transcribe(
            video_path,
            beam_size=5,
            condition_on_previous_text=False 
        )
        
        transcribed_segments = []
        for segment in segments_generator:
            transcribed_segments.append(
                AudioSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=1.0
                )
            )

        self.logger.info(f"Transcription complete: {len(transcribed_segments)} segments")
        return transcribed_segments
