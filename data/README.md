# Dataset

Run `python scripts/download_dataset.py` from the repository root to download
10,000 drawings per configured class and save them here. Set `--count` to
choose a different cap, or pass `--overwrite` to replace existing archives.
For example: `python scripts/download_dataset.py --count 20000`.

Each generated archive has an `images` array with shape `N×784` (flattened
28×28 grayscale bitmap) and integer values in 0–255. Files use the class name
from `config/classes.txt`, with spaces replaced by underscores (for example
`ice_cream.npz`).

The loader reads archives one at a time and caps examples per class. Obtain
data from the official Quick, Draw! dataset source and review its terms before
redistributing it.
