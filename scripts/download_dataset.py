"""Download a balanced sample of Google's public Quick, Draw! bitmap data."""
import argparse
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def read_range(url: str, start: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"Server did not honor byte range for {url}: HTTP {response.status}")
        return response.read()


def read_header(prefix: bytes) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    stream = BytesIO(prefix)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    return shape, fortran_order, dtype, stream.tell()


def download_class(name: str, count: int, output_dir: Path, overwrite: bool) -> None:
    filename = name.replace(" ", "_") + ".npz"
    output_path = output_dir / filename
    if output_path.exists() and not overwrite:
        with np.load(output_path) as existing:
            if "images" in existing and len(existing["images"]) >= count:
                print(f"{name}: already present ({len(existing['images']):,}), skipping")
                return

    url = BASE_URL + quote(name, safe="") + ".npy"
    shape, fortran_order, dtype, offset = read_header(read_range(url, 0, 65535))
    if len(shape) != 2 or shape[1] != 784:
        raise ValueError(f"Unexpected bitmap array shape for {name}: {shape}")
    take = min(count, shape[0])
    payload = read_range(url, 0, offset + take * 784 - 1)
    # Parse the header again from the exact response, then view only the requested
    # rows. The Google bitmap arrays are uint8 and C-contiguous.
    actual_shape, fortran_order, dtype, offset = read_header(payload[:65536])
    images = np.frombuffer(payload, dtype=dtype, count=take * 784, offset=offset).copy()
    images = images.reshape(take, 784)
    if fortran_order:
        images = np.ascontiguousarray(images)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, images=images)
    temporary.replace(output_path)
    print(f"{name}: saved {take:,} of {shape[0]:,} drawings to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default="config/classes.txt")
    parser.add_argument("--output", default="data")
    parser.add_argument("--count", type=int, default=10000,
                        help="maximum drawings per class (default: 10000)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    classes = [line.strip() for line in Path(args.classes).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for name in classes:
        try:
            download_class(name, args.count, Path(args.output), args.overwrite)
        except (HTTPError, URLError) as exc:
            raise SystemExit(f"Download failed for {name}: {exc}") from exc


if __name__ == "__main__":
    main()
