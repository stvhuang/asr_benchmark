import argparse
import datetime
import json
import re
import time
from pathlib import Path

import datasets
import jiwer
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def get_processor_and_model(
    model_name_or_path: str,
    device: torch.device,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:

    processor = WhisperProcessor.from_pretrained(model_name_or_path)

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    )
    model = model.to(device=device)
    model.eval()

    return processor, model


def normalize_text(
    text: str,
) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # collapse whitespace

    return text


def transcribe_with_whisper(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    audio_arrays: list[np.ndarray],
    sampling_rate: int,
    device: torch.device,
) -> tuple[list[str], float]:
    """Transcribe a batch of audio arrays.

    Args:
        audio_arrays: List of 1-D numpy arrays (one per utterance).
        sampling_rate: Sampling rate shared by all arrays.

    Returns:
        (hypotheses, latency) – list of transcription strings and the
        wall-clock inference time in seconds for the whole batch.
    """
    input_features = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features
    input_features = input_features.to(device=device, dtype=model.dtype)

    t_start = time.perf_counter()

    with torch.no_grad():
        predicted_token_ids = model.generate(
            input_features,
            task="transcribe",
        )

    # synchronize if on GPU/MPS to get accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "mps":
        torch.mps.synchronize()

    lat = time.perf_counter() - t_start

    hyps = processor.batch_decode(predicted_token_ids, skip_special_tokens=True)

    return hyps, lat


def transcribe_with_meralion():
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda", "mps"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        "-m",
        choices=["openai/whisper-large-v3", "MERaLiON/MERaLiON-2-10B-ASR"],
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", "-b", default=1, type=int, required=True)
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load processor and model
    logger.info(f"Loading model and processor from {args.model_name_or_path} ...")
    processor, model = get_processor_and_model(
        args.model_name_or_path,
        device=device,
    )

    # Load dataset
    print("Loading FLEURS en_us test split ...")
    dataset = datasets.load_dataset(
        path="google/fleurs",
        name="en_us",
        split="test",
        revision="refs/pr/31",  # https://huggingface.co/datasets/google/fleurs/discussions/31
    )

    if args.num_samples is not None:
        logger.info(f"Selecting first {args.num_samples} samples for benchmarking ...")
        dataset = dataset.select(range(args.num_samples))

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Run benchmark
    refs = []
    hyps = []
    lats = []
    durs = []

    num_samples = len(dataset)
    logger.info(f"Running benchmark (batch_size: {args.batch_size}) ...")

    for batch_start in tqdm(range(0, num_samples, args.batch_size)):
        batch_end = min(batch_start + args.batch_size, num_samples)
        batch_slice = dataset.select(range(batch_start, batch_end))

        batch_audio_arrays = []
        batch_refs = []
        batch_durs = []
        sample_rate = None

        for sample in batch_slice:
            audio_samples = sample["audio"].get_all_samples()
            audio_array = audio_samples.data.squeeze(0).numpy()  # (N,)
            sample_rate = audio_samples.sample_rate

            batch_audio_arrays.append(audio_array)
            batch_refs.append(normalize_text(sample["raw_transcription"]))
            batch_durs.append(audio_array.shape[0] / sample_rate)

        batch_hyps, lat = transcribe_with_whisper(
            processor,
            model,
            batch_audio_arrays,
            sample_rate,
            device,
        )

        per_sample_lat = lat / len(batch_audio_arrays)

        for j, (ref, hyp) in enumerate(zip(batch_refs, batch_hyps)):
            hyp = normalize_text(hyp)
            hyps.append(hyp)
            refs.append(batch_refs[j])
            lats.append(per_sample_lat)
            durs.append(batch_durs[j])

            if args.debug:
                logger.info(f"hyp: {hyp}")
                logger.info(f"ref: {ref}")

    # Compute metrics
    total_dur = sum(durs)
    total_lat = sum(lats)

    results = {
        "args": vars(args),
        "performance": {
            "cer": round(jiwer.cer(refs, hyps), 4),
            "mer": round(jiwer.mer(refs, hyps), 4),
            "wer": round(jiwer.wer(refs, hyps), 4),
        },
        "latency": {
            "mean": np.mean(lats).round(4),
            "median": np.median(lats).round(4),
            "p90": np.percentile(lats, 90).round(4),
            "p99": np.percentile(lats, 99).round(4),
            "min": np.min(lats).round(4),
            "max": np.max(lats).round(4),
        },
        "throughput": {
            "total_audio_duration_in_seconds": round(total_dur, 4),
            "total_inference_time_in_seconds": round(total_lat, 4),
            "inference_time_to_audio_duration_ratio": round(total_lat / total_dur, 4),
        },
    }

    logger.info("\n" + json.dumps(results, indent=2))

    results["raw_results"] = {
        "refs": refs,
        "hyps": hyps,
        "lats": lats,
        "durs": durs,
    }

    # Save results to JSON file with timestamp
    results_savepath = Path(
        f"./results/{args.model_name_or_path.replace('/', '-')}_{args.device}_bs{args.batch_size}_{datetime.datetime.now():%Y%m%dT%H%M%S}.json"
    )
    results_savepath.parent.mkdir(parents=True, exist_ok=True)

    with open(results_savepath, "w") as f:
        json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_savepath}.")
