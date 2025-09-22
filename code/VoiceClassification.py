import subprocess
from pathlib import Path
import ast
import json
import os
import logging
import torch
import whisper
import google.generativeai as genai
from typing import List, Optional

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SpeechRecog")


class SpeechRecognitionForClassification:
    """
    A pipeline for speech transcription, refinement, and classification.

    Features:
    - Load and transcribe audio files using Whisper (GPU/CPU) or Whisper.cpp (CLI).
    - Refine raw ASR outputs with Gemini for better readability.
    - Classify refined text into user-defined categories using Gemini.
    - Manage JSON mappings and transcription storage for datasets.

    Parameters
    ----------
    audio_folder : str
        Path to the folder containing audio files.
    classes : list[str], optional
        List of categories for classification.
    audio_mapping_file : str, optional
        Path to JSON mapping file for merging transcriptions with metadata.
    gemini_api_key : str, optional
        Google Gemini API key (required for refinement and classification).
    backend : str, default="whisper"
        Transcription backend: "whisper" (PyTorch) or "whispercpp" (CLI).
    """

    def __init__(self,
                 audio_folder: str,
                 classes: Optional[List[str]] = None,
                 audio_mapping_file: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 backend: str = "whisper"):  
        self.AUDIO_FOLDER = audio_folder
        self.JSON_AUDIO_MAPPING = audio_mapping_file or "json_audio_mapping.json"
        self.classes = classes or []
        self.OUTPUT_TRANSCRIPTIONS: Optional[str] = None
        self.gemini_api_key = gemini_api_key
        self.backend = backend
        logger.info("Initialized with folder=%s mapping=%s classes=%s backend=%s",
                    audio_folder, self.JSON_AUDIO_MAPPING, self.classes, backend)

    # ---------- Gemini ----------
    def configure_gemini(self):
        """Configure Gemini API with the provided key."""
        if not self.gemini_api_key:
            logger.error("Gemini API key is missing")
            raise ValueError("Gemini API key is missing.")
        genai.configure(api_key=self.gemini_api_key)
        logger.info("Gemini configured")

    def load_gemini_model(self, model_name="gemini-2.0-flash", temperature=0.5):
        """Load a Gemini model with given configuration."""
        try:
            self.configure_gemini()
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": temperature},
            )
            logger.info("Loaded Gemini model: %s", model_name)
            return model
        except Exception:
            logger.exception("Failed to load Gemini model")
            raise

    # ---------- Classification ----------
    def classify_text(self, text: str, language: str = "algerian") -> List[str]:
        """
        Classify text into predefined categories using Gemini.

        Parameters
        ----------
        text : str
            Input text for classification.
        language : str, default="algerian"
            Language description for prompt context.

        Returns
        -------
        list[str]
            Predicted classes.
        """
        prompt = (
            f"The following {language} text may include more than one speaker. "
            f"Classify it into: {self.classes}. "
            f"Allow multiple classes and return only a Python list "
            f"(response must start with [ and end with ]). \nText:\n{text}"
        )
        try:
            model = self.load_gemini_model()
            raw = model.generate_content(prompt).text
            logger.info("Classification response: %s", raw)
            return ast.literal_eval(raw)
        except Exception:
            logger.exception("Error during text classification")
            raise

    def classify_audio(self, audio_path: str) -> List[str]:
        """
        Transcribe, refine, and classify a single audio file.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file.

        Returns
        -------
        list[str]
            Predicted classes.
        """
        try:
            text = self.generate_transcription(audio_file=audio_path)["text"]
            refined = self.generate_refinement(text)
            return self.classify_text(refined)
        except Exception:
            logger.exception("Failed to classify audio: %s", audio_path)
            raise

    # ---------- Audio / Whisper ----------
    def load_audio_files(self) -> List[str]:
        """Return list of supported audio files (.mp3, .wav, .webm) in folder."""
        try:
            if not os.path.exists(self.AUDIO_FOLDER):
                raise FileNotFoundError(f"No folder {self.AUDIO_FOLDER}")
            files = [
                f for f in os.listdir(self.AUDIO_FOLDER)
                if f.lower().endswith((".mp3", ".wav", ".webm"))
            ]
            logger.info("Found %d audio files", len(files))
            return files
        except Exception:
            logger.exception("Error loading audio files")
            raise

    def get_transcription_model(self):
        """Load Whisper model (PyTorch)."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("medium", device=device)
            logger.info("Loaded Whisper model on %s", device)
            return model
        except Exception:
            logger.exception("Failed to load Whisper model")
            raise

    def generate_transcription(self,
                               audio_file: str,
                               use_gpu=False,
                               model=None,
                               exe_path=r"C:\Users\ACER\whisper.cpp\build\bin\Release\whisper-cli.exe",
                               model_path=r"C:\Users\ACER\whisper.cpp\models\ggml-medium.bin",
                               threads=4,
                               language="ar"):
        """
        Transcribe an audio file using Whisper or Whisper.cpp.

        Parameters
        ----------
        audio_file : str
            Filename of the audio file inside self.AUDIO_FOLDER.
        use_gpu : bool, default=False
            Use GPU acceleration with Whisper (ignored by whisper.cpp).
        model : whisper.Whisper, optional
            Preloaded Whisper model (for efficiency).
        exe_path : str
            Path to whisper-cli.exe (whisper.cpp).
        model_path : str
            Path to whisper.cpp model file.
        threads : int, default=4
            Number of threads for whisper.cpp.
        language : str, default="ar"
            Language code for transcription.

        Returns
        -------
        dict
            {"text": transcription text}
        """
        full_path = os.path.join(self.AUDIO_FOLDER, audio_file)

        if self.backend == "whisper":
            try:
                model = model or self.get_transcription_model()
                logger.info("Transcribing %s with Whisper", full_path)
                return model.transcribe(full_path, language=language, fp16=use_gpu)
            except Exception:
                logger.exception("Transcription failed for %s", audio_file)
                raise

        elif self.backend == "whispercpp":
            try:
                logger.info("Transcribing %s with Whisper.cpp", full_path)
                cmd = [
                    exe_path,
                    "-m", model_path,
                    "-f", str(full_path),
                    "-l", language,
                    "-t", str(threads)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return {"text": result.stdout.strip()}
            except Exception:
                logger.exception("Whisper.cpp transcription failed for %s", audio_file)
                raise

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def generate_transcriptions(self, audio_files: List[str], **kwargs):
        """
        Transcribe a list of audio files.

        Parameters
        ----------
        audio_files : list[str]
            Filenames to transcribe.
        **kwargs : dict
            Passed to generate_transcription.

        Returns
        -------
        list[dict]
            [{"audio": filename, "transcription": text}, ...]
        """
        results = []
        model = None if self.backend == "whispercpp" else self.get_transcription_model()
        for f in audio_files:
            try:
                res = self.generate_transcription(audio_file=f, model=model, **kwargs)
                results.append({"transcription": res["text"], "audio": f})
            except Exception:
                logger.exception("Failed to transcribe %s", f)
        return results

    # ---------- JSON helpers ----------
    def read_json_file(self, path: str):
        """Load a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded JSON: %s", path)
            return data
        except Exception:
            logger.exception("Failed reading JSON: %s", path)
            raise

    def save_to_json(self, data, path: str):
        """Save data to a JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved JSON: %s", path)
        except Exception:
            logger.exception("Failed saving JSON: %s", path)
            raise

    def save_transcriptions(self, transcriptions_list, output_transcriptions_file: str):
        """Save transcription results to file and register path for later merging."""
        self.OUTPUT_TRANSCRIPTIONS = output_transcriptions_file
        self.save_to_json(transcriptions_list, output_transcriptions_file)

    def add_transcription_to_original_file(self):
        """
        Merge saved transcriptions into the original mapping JSON
        under the key 'whisper_transcription'.
        """
        if not self.OUTPUT_TRANSCRIPTIONS:
            logger.error("No output transcription file set.")
            raise ValueError("No output transcription file set.")
        try:
            mapping_data = self.read_json_file(self.JSON_AUDIO_MAPPING)
            trans_data = self.read_json_file(self.OUTPUT_TRANSCRIPTIONS)

            for obj in trans_data:
                name_no_ext, _ = os.path.splitext(obj["audio"])
                matched = False
                for row in mapping_data:
                    if row.get("audio") == name_no_ext:
                        row["whisper_transcription"] = obj["transcription"]
                        matched = True
                        break
                if not matched:
                    logger.warning("Item %s not found in mapping!", obj["audio"])

            self.save_to_json(mapping_data, self.JSON_AUDIO_MAPPING)
            logger.info("Merged transcriptions into mapping file")
        except Exception:
            logger.exception("Failed to merge transcriptions")
            raise

    # ---------- Text refinement ----------
    def generate_refinement(self, text: str) -> str:
        """
        Refine raw ASR output using Gemini to fix spelling and context
        without altering meaning or structure.

        Parameters
        ----------
        text : str
            ASR transcription text.

        Returns
        -------
        str
            Refined text.
        """
        refinement_prompt = """ You are an expert in Algerian Arabic (Darja).  
            Your task is to proofread the following text.  

            - Correct spelling mistakes
            - Fix mis-segmented words
            - Correct place names if possible
            - Do not add/remove/change meaning
            - Return only the corrected text. Here's the text: """
        prompt = refinement_prompt + text
        try:
            model = self.load_gemini_model()
            refined = model.generate_content(prompt).text
            logger.info("Refinement complete")
            return refined
        except Exception:
            logger.exception("Text refinement failed")
            raise

    # ---------- Classify a folder ----------
    def classify_folder(self, threads=4) -> list:
        """
        Transcribe, refine, and classify all audio files in folder.

        Parameters
        ----------
        threads : int, default=4
            Number of threads for whisper.cpp backend.

        Returns
        -------
        list[dict]
            [{"audio": filename, "classes": [...]}, ...]
        """
        results = []
        try:
            files = self.load_audio_files()
            for f in files:
                logger.info("Classifying %s", f)
                try:
                    text = self.generate_transcription(audio_file=f, threads=threads)["text"]
                    refined = self.generate_refinement(text)
                    classes = self.classify_text(refined)
                    results.append({"audio": f, "classes": classes})
                except Exception:
                    logger.exception("Failed to classify %s", f)
        except Exception:
            logger.exception("Could not process folder %s", self.AUDIO_FOLDER)
            raise
        return results
