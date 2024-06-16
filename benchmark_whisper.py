import numpy as np
from omegaconf import OmegaConf
import torch
import torch.backends.cudnn as cudnn
from transformers import pipeline, AutoProcessor
from transformers.modeling_utils import is_flash_attn_2_available

import os
import sys
import time
import json
import random
import logging
import argparse
import subprocess

from logging_utils import setup_logging

logger = logging.getLogger("benchmark")


def get_arg_parser(description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        help="Attention implementation to use: 'flash_attention_2' or 'sdpa'.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision.",
    )
    parser.add_argument(
        "--chunk-length-s",
        type=int,
        help="Length of audio chunks to process.",
        default=30,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Whisper model name.",
    )

    return parser


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def setup(args):
    cudnn.benchmark = True
    cfg = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None})
    if args.config_file:
        cfg = OmegaConf.merge(OmegaConf.load(args.config_file), cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    seed = getattr(args, "seed", 0)
    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("benchmark")

    fix_random_seeds(seed)

    cfg.pop("config_file")
    write_config(cfg, args.output_dir)

    return cfg


def load_audio_file(audio_file, model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    sr = processor.feature_extractor.sampling_rate
    x = ffmpeg_read(audio_file, sr)
    audio_duration = len(x) // sr
    logger.info(f"Length of the interview in seconds: {audio_duration}")

    return x, audio_duration


def ffmpeg_read(file_path, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    with open(file_path, "rb") as f:
        bpayload = f.read()
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError(
            "ffmpeg was not found but is required to load audio files from filename"
        ) from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


def load_model(args):
    logger.info(f"Loading model: {args.model_name}")

    pipe = pipeline(
        "automatic-speech-recognition",
        args.model_name,
        device="cuda",
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        model_kwargs={"attn_implementation": args.attn_implementation},
    )
    return pipe


def main(args):
    cfg = setup(args)

    total_time = time.time()
    audio, audio_duration = load_audio_file(cfg.audio_file, cfg.model_name)
    model = load_model(cfg)

    transcription_time = time.time()
    outputs = model(
        audio,
        chunk_length_s=cfg.chunk_length_s,
        batch_size=cfg.batch_size,
        return_timestamps=True,
        generate_kwargs={"task": "translate"},
    )
    transcription_time = time.time() - transcription_time
    total_time = time.time() - total_time
    seconds_transcribed_per_second = audio_duration / transcription_time

    logger.info(f"Transcription time: {int(transcription_time)} seconds")
    logger.info(f"Seconds transcribed per second: {seconds_transcribed_per_second}")
    logger.info(f"Total time elapsed: {int(total_time)} seconds")

    outputs["audio_duration"] = audio_duration
    outputs["processing_time"] = transcription_time
    outputs["seconds_transcribed_per_second"] = seconds_transcribed_per_second

    # save transcription
    with open(os.path.join(args.output_dir, "outputs.json"), "w") as json_file:
        json.dump(outputs, json_file, indent=4)


if __name__ == "__main__":
    description = "Benchmark whisper configurations"
    args_parser = get_arg_parser(description=description)
    args = args_parser.parse_args()

    assert (
        not args.attn_implementation == "flash_attention_2" or is_flash_attn_2_available()
    ), "Flash attention 2 is not available. Please install with 'pip install flash-attn --no-build-isolation'."

    sys.exit(main(args))
