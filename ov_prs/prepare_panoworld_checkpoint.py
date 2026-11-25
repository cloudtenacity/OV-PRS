"""Normalize legacy PanoWorld-Hstar Safetensors keys for the Transformers loader."""

import argparse
import json
import shutil
import struct
import tempfile
from pathlib import Path

PREFIX_MAP = {
    "model.language_model.language_model.language_model.": "model.language_model.",
    "model.language_model.visual.": "model.visual.",
}
LEGACY_PREFIXES = tuple(PREFIX_MAP)
INDEX_NAME = "model.safetensors.index.json"


def _current_key(key):
    for old, new in PREFIX_MAP.items():
        if key.startswith(old):
            return new + key[len(old):]
    return key


def _convert_keys(entries):
    converted = {_current_key(key): value for key, value in entries.items()}
    if len(converted) != len(entries):
        raise ValueError("Checkpoint contains conflicting legacy and current tensor names.")
    return converted


def weight_files(directory):
    single_file = directory / "model.safetensors"
    if single_file.is_file():
        return [single_file]

    index_file = directory / INDEX_NAME
    if index_file.is_file():
        index = json.loads(index_file.read_text())
        shards = sorted(set(index["weight_map"].values()))
        if shards and all(Path(name).name == name for name in shards):
            return [directory / name for name in shards]

    raise FileNotFoundError(f"No Safetensors model found in {directory}.")


def _copy_with_current_keys(source, destination):
    with source.open("rb") as original:
        header_size, = struct.unpack("<Q", original.read(8))
        if not 2 <= header_size <= 100_000_000:
            raise ValueError(f"Invalid Safetensors header in {source}.")
        entries = json.loads(original.read(header_size))
        converted = _convert_keys(entries)
        if entries.keys() == converted.keys():
            shutil.copy2(source, destination)
            return 0

        # The offsets in the header are relative to the tensor buffer.
        # Renaming keys leaves that buffer and every offset intact.
        header = json.dumps(converted, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        with destination.open("wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            shutil.copyfileobj(original, output, length=8 * 1024 * 1024)
    return sum(_current_key(key) != key for key in entries)


def prepare_checkpoint(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError("Choose a new destination outside the source checkpoint.")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"No model configuration found in {source}.")

    shards = weight_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".panoworld-", dir=destination.parent) as temp:
        converted = Path(temp) / "checkpoint"
        converted.mkdir()
        renamed = sum(_copy_with_current_keys(shard, converted / shard.name) for shard in shards)

        for file in source.iterdir():
            if file.name == INDEX_NAME and shards[0].name != "model.safetensors":
                index = json.loads(file.read_text())
                index["weight_map"] = _convert_keys(index["weight_map"])
                (converted / file.name).write_text(json.dumps(index, indent=2) + "\n")
            elif (file.is_file() and file.suffix in {".json", ".jinja", ".model", ".tiktoken", ".txt"}
                  and file.name not in {INDEX_NAME, "trainer_state.json"}):
                shutil.copy2(file, converted / file.name)

        converted.rename(destination)
    return renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded PanoWorld-Hstar checkpoint")
    parser.add_argument("destination", type=Path, help="New directory for the compatible checkpoint")
    args = parser.parse_args()
    renamed = prepare_checkpoint(args.source, args.destination)
    print(f"Prepared {args.destination} ({renamed} tensor names updated).")


if __name__ == "__main__":
    main()
