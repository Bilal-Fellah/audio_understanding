import subprocess
from pathlib import Path
import ast
import json
import os
import logging
import torch
import whisper
import google.generativeai as genai
from typing import List, Optional, Dict, Any

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SpeechRecog")


class VoiceTopicExtraction:
    """
    A pipeline for speech transcription, refinement, and classification.

    This class provides a complete workflow for:
    - Transcribing audio files using Whisper (GPU/CPU)
    - Refining transcriptions with Google Gemini AI
    - Classifying transcriptions into predefined categories
    - Managing transcription history in JSON format

    Attributes
    ----------
    audio_folder : str
        Path to the folder containing audio files
    classes : List[str]
        List of categories for text classification
    audio_mapping_file : str
        Path to JSON file storing transcription history
    gemini_api_key : str
        Google Gemini API key for AI operations
    backend : str
        Transcription backend identifier (for reference only)

    Example
    -------
    >>> extractor = VoiceTopicExtraction(
    ...     audio_folder="./audio_samples",
    ...     classes=["politics", "sports", "technology"],
    ...     gemini_api_key="your-api-key"
    ... )
    >>> results = extractor.classify_folder(device="cuda")
    """

    def __init__(
        self,
        audio_folder: str,
        classes: Optional[List[str]] = None,
        audio_mapping_file: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        backend: str = "whisper"
    ):
        """
        Initialize the VoiceTopicExtraction pipeline.

        Parameters
        ----------
        audio_folder : str
            Path to directory containing audio files (.mp3, .wav, .webm)
        classes : List[str], optional
            Categories for classification (default: empty list)
        audio_mapping_file : str, optional
            JSON file path for history (default: "json_audio_mapping.json")
        gemini_api_key : str, optional
            Google Gemini API key (can be set later via configure_gemini)
        backend : str, optional
            Backend identifier for logging (default: "whisper")
        """
        self.AUDIO_FOLDER = audio_folder
        self.JSON_AUDIO_MAPPING = audio_mapping_file or "json_audio_mapping.json"
        self.classes = classes or []
        self.gemini_api_key = gemini_api_key
        self.backend = backend
        
        logger.info(
            "Initialized with folder=%s mapping=%s classes=%s backend=%s",
            audio_folder, self.JSON_AUDIO_MAPPING, self.classes, backend
        )

    # ---------- Gemini Configuration ----------
    
    def configure_gemini(self, api_key: Optional[str] = None) -> None:
        """
        Configure Google Gemini API.

        Parameters
        ----------
        api_key : str, optional
            API key to use (if not provided, uses instance key)

        Raises
        ------
        ValueError
            If no API key is available
        """
        if api_key:
            self.gemini_api_key = api_key

        if not self.gemini_api_key:
            logger.error("Gemini API key is missing")
            raise ValueError("Gemini API key is required for refinement and classification")
        
        genai.configure(api_key=self.gemini_api_key)
        logger.info("Gemini API configured successfully")

    def load_gemini_model(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5
    ) -> genai.GenerativeModel:
        """
        Load a Google Gemini generative model.

        Parameters
        ----------
        model_name : str, optional
            Gemini model identifier (default: "gemini-2.0-flash")
        temperature : float, optional
            Sampling temperature 0.0-1.0 (default: 0.5)

        Returns
        -------
        genai.GenerativeModel
            Configured Gemini model instance

        Raises
        ------
        Exception
            If model loading fails
        """
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

    # ---------- Text Classification ----------
    
    def classify_text(self, text: str, language: str = "Algerian Arabic") -> List[str]:
        """
        Classify text into predefined categories using Gemini.

        Parameters
        ----------
        text : str
            Text to classify
        language : str, optional
            Language description for context (default: "Algerian Arabic")

        Returns
        -------
        List[str]
            List of predicted category labels

        Raises
        ------
        Exception
            If classification fails or response parsing fails
        """
        prompt = (
            f"The following {language} text may include multiple speakers. "
            f"Analyze the general topics and classify this text into these categories: {self.classes}. "
            f"Multiple categories are allowed. Return ONLY a Python list in the format: ['category1', 'category2']. "
            f"Your response must start with [ and end with ].\n\nText:\n{text}"
        )
        try:
            model = self.load_gemini_model()
            raw = model.generate_content(prompt).text.strip()
            logger.info("Classification response: %s", raw)
            
            # Extract list if wrapped in markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("python"):
                    raw = raw[6:].strip()
            
            return ast.literal_eval(raw)
        except Exception:
            logger.exception("Error during text classification")
            raise

    def classify_audio(
        self,
        audio_path: str,
        device: str = "cpu",
        threads: int = 4,
        transcription_model: Optional[str] = None,
        exe_path: str = r"C:\Users\ACER\whisper.cpp\build\bin\Release\whisper-cli.exe",
        model_path: str = r"C:\Users\ACER\whisper.cpp\models\ggml-medium.bin"
    ) -> List[str]:
        """
        Transcribe, refine, and classify a single audio file.

        This method executes the full pipeline:
        1. Transcribe audio to text
        2. Refine transcription with Gemini
        3. Classify refined text
        4. Save results to history file

        Parameters
        ----------
        audio_path : str
            Audio filename (relative to audio_folder)
        device : str, optional
            "cuda" for GPU or "cpu" for CPU processing (default: "cpu")
        threads : int, optional
            CPU threads for whisper.cpp (default: 4)
        transcription_model : str, optional
            Whisper model size for GPU (default: "medium")
        exe_path : str, optional
            Path to whisper-cli.exe for CPU processing
        model_path : str, optional
            Path to whisper.cpp model file for CPU processing

        Returns
        -------
        List[str]
            Predicted category labels

        Raises
        ------
        Exception
            If any pipeline step fails
        """
        try:
            # Set default model if not specified
            if transcription_model is None:
                transcription_model = "medium"
            
            # Transcribe audio
            transcription_result = self.generate_transcription(
                audio_file=audio_path,
                device=device,
                threads=threads,
                model_name=transcription_model,
                exe_path=exe_path,
                model_path=model_path
            )
            text = transcription_result["text"]
            
            # Refine transcription
            refined = self.generate_refinement(text)
            
            # Classify text
            classes = self.classify_text(refined)
            
            # Save to history
            history_data = self.read_json_file(self.JSON_AUDIO_MAPPING) or []
            history_data.append({
                "audio": audio_path,
                "transcription": text,
                "transcription_refined": refined,
                "classes": classes,
                "transcription_engine": f"{device} - {transcription_model}"
            })
            self.save_to_json(history_data, self.JSON_AUDIO_MAPPING)
            
            return classes
        except Exception:
            logger.exception("Failed to classify audio: %s", audio_path)
            raise

    # ---------- Audio File Management ----------
    
    def load_audio_files(self, folder_path: Optional[str] = None) -> List[str]:
        """
        List all supported audio files in the specified folder.

        Parameters
        ----------
        folder_path : str, optional
            Override default audio folder (updates instance folder)

        Returns
        -------
        List[str]
            List of audio filenames (.mp3, .wav, .webm)

        Raises
        ------
        FileNotFoundError
            If folder doesn't exist
        """
        try:
            if folder_path:
                self.AUDIO_FOLDER = folder_path
            
            if not os.path.exists(self.AUDIO_FOLDER):
                raise FileNotFoundError(f"Audio folder not found: {self.AUDIO_FOLDER}")
            
            files = [
                f for f in os.listdir(self.AUDIO_FOLDER)
                if f.lower().endswith((".mp3", ".wav", ".webm"))
            ]
            logger.info("Found %d audio files in %s", len(files), self.AUDIO_FOLDER)
            return files
        except Exception:
            logger.exception("Error loading audio files")
            raise

    # ---------- Whisper Transcription ----------
    
    def get_transcription_model(
        self,
        model_name: str = "medium",
        device: Optional[str] = None
    ) -> whisper.Whisper:
        """
        Load a Whisper model for PyTorch-based transcription.

        Parameters
        ----------
        model_name : str, optional
            Whisper model size: tiny, base, small, medium, large (default: "medium")
        device : str, optional
            "cuda" or "cpu" (auto-detected if not specified)

        Returns
        -------
        whisper.Whisper
            Loaded Whisper model instance

        Raises
        ------
        Exception
            If model loading fails
        """
        try:
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model = whisper.load_model(model_name, device=device)
            logger.info("Loaded Whisper model '%s' on %s", model_name, device)
            return model
        except Exception:
            logger.exception("Failed to load Whisper model")
            raise

    def generate_transcription(
        self,
        audio_file: str,
        device: str = "cpu",
        model_name: str = "medium",
        exe_path: str = r"C:\Users\ACER\whisper.cpp\build\bin\Release\whisper-cli.exe",
        model_path: str = r"C:\Users\ACER\whisper.cpp\models\ggml-medium.bin",
        threads: int = 4,
        language: str = "ar"
    ) -> Dict[str, str]:
        """
        Transcribe an audio file using Whisper (GPU) or whisper.cpp (CPU).

        Parameters
        ----------
        audio_file : str
            Audio filename relative to audio_folder
        device : str, optional
            "cuda" for GPU or "cpu" for CPU (default: "cpu")
        model_name : str, optional
            Whisper model size for GPU mode (default: "medium")
        exe_path : str, optional
            Path to whisper-cli.exe for CPU mode
        model_path : str, optional
            Path to whisper.cpp model for CPU mode
        threads : int, optional
            Number of CPU threads for whisper.cpp (default: 4)
        language : str, optional
            Language code (e.g., "ar" for Arabic, "en" for English) (default: "ar")

        Returns
        -------
        Dict[str, str]
            Dictionary with "text" key containing transcription

        Raises
        ------
        ValueError
            If device is not "cuda" or "cpu"
        Exception
            If transcription fails
        """
        full_path = os.path.join(self.AUDIO_FOLDER, audio_file)

        if device == "cuda":  # GPU transcription
            try:
                model = self.get_transcription_model(model_name=model_name, device=device)
                logger.info("Transcribing %s with Whisper (GPU)", full_path)
                return model.transcribe(full_path, language=language, fp16=True)
            except Exception:
                logger.exception("GPU transcription failed for %s", audio_file)
                raise

        elif device == "cpu":  # CPU transcription via whisper.cpp
            try:
                logger.info("Transcribing %s with whisper.cpp (CPU)", full_path)
                cmd = [
                    exe_path,
                    "-m", model_path,
                    "-f", str(full_path),
                    "-l", language,
                    "-t", str(threads)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return {"text": result.stdout.strip()}
            except subprocess.CalledProcessError as e:
                logger.error("whisper.cpp failed: %s", e.stderr)
                raise
            except Exception:
                logger.exception("whisper.cpp transcription failed for %s", audio_file)
                raise

        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cuda' or 'cpu'")

    # ---------- JSON Utilities ----------
    
    def read_json_file(self, path: str) -> Any:
        """
        Load and parse a JSON file.

        Parameters
        ----------
        path : str
            Path to JSON file

        Returns
        -------
        Any
            Parsed JSON data, or empty list if file doesn't exist

        Raises
        ------
        Exception
            If file reading or JSON parsing fails
        """
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded JSON from: %s", path)
                return data
            else:
                logger.info("JSON file not found, returning empty list: %s", path)
                return []
        except Exception:
            logger.exception("Failed to read JSON file: %s", path)
            raise

    def save_to_json(self, data: Any, path: str) -> None:
        """
        Save data to a JSON file with UTF-8 encoding.

        Parameters
        ----------
        data : Any
            Data to serialize to JSON
        path : str
            Output file path

        Raises
        ------
        Exception
            If file writing fails
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved JSON to: %s", path)
        except Exception:
            logger.exception("Failed to save JSON file: %s", path)
            raise

    # ---------- Text Refinement ----------
    
    def generate_refinement(
        self,
        text: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.5
    ) -> str:
        """
        Refine raw ASR transcription using Gemini AI.

        Corrects spelling, word segmentation, and place names while
        preserving the original meaning and structure.

        Parameters
        ----------
        text : str
            Raw ASR transcription text
        model : str, optional
            Gemini model to use (default: "gemini-2.0-flash")
        temperature : float, optional
            Sampling temperature (default: 0.5)

        Returns
        -------
        str
            Refined, corrected transcription

        Raises
        ------
        Exception
            If refinement request fails
        """
        refinement_prompt = """You are an expert in Algerian Arabic (Darja).  
Your task is to proofread the following text.  

- Correct spelling mistakes
- Fix mis-segmented words
- Correct place names if possible
- Do NOT add, remove, or change the meaning
- Return ONLY the corrected text without any explanations

Here's the text: """
        
        prompt = refinement_prompt + text
        
        try:
            gemini_model = self.load_gemini_model(model_name=model, temperature=temperature)
            refined = gemini_model.generate_content(prompt).text.strip()
            logger.info("Text refinement complete")
            return refined
        except Exception:
            logger.exception("Text refinement failed")
            raise

    # ---------- Batch Processing ----------
    
    def classify_folder(
        self,
        folder_path: Optional[str] = None,
        device: str = "cpu",
        transcription_model: Optional[str] = None,
        threads: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process all audio files in a folder through the complete pipeline.

        This method:
        1. Discovers all audio files in the folder
        2. Transcribes each file
        3. Refines each transcription
        4. Classifies each refined text
        5. Saves all results to history

        Parameters
        ----------
        folder_path : str, optional
            Override default audio folder
        device : str, optional
            "cuda" or "cpu" (default: "cpu")
        transcription_model : str, optional
            Whisper model size for GPU (default: "medium")
        threads : int, optional
            CPU threads for whisper.cpp (default: 4)

        Returns
        -------
        List[Dict[str, Any]]
            List of results: [{"audio": filename, "classes": [...]}, ...]

        Raises
        ------
        Exception
            If folder processing fails
        """
        results = []
        
        try:
            # Set default model
            if transcription_model is None:
                transcription_model = "medium"
            
            files = self.load_audio_files(folder_path=folder_path)
            
            for idx, audio_file in enumerate(files, 1):
                logger.info("Processing file %d/%d: %s", idx, len(files), audio_file)
                try:
                    classes = self.classify_audio(
                        audio_file,
                        device=device,
                        transcription_model=transcription_model,
                        threads=threads
                    )
                    results.append({"audio": audio_file, "classes": classes})
                    logger.info("Successfully classified %s: %s", audio_file, classes)
                except Exception:
                    logger.exception("Failed to classify %s, skipping...", audio_file)
                    # Continue with next file instead of stopping
                    continue
            
            logger.info("Folder processing complete. Processed %d/%d files successfully",
                       len(results), len(files))
        except Exception:
            logger.exception("Could not process folder %s", self.AUDIO_FOLDER)
            raise
        
        return results