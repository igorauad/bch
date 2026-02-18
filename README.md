# bch

A small C++ implementation of a BCH (Bose–Chaudhuri–Hocquenghem) codec.

This library provides BCH encoding and decoding for binary linear block codes over GF(2^m). It was originally developed for use in [gr-dvbs2rx](https://github.com/igorauad/gr-dvbs2rx), a GNU Radio DVB-S2 receiver implementation.

## Features
- BCH encoder
- BCH decoder (error correction)
- Finite field (GF) utilities
- Lightweight and dependency-free (library only)
- MIT licensed
- Python implementation for experimentation and learning

## Build

```bash
cmake -S . -B build -DBCH_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Install

```bash
cmake --install build
```

Headers are installed under:

```
include/bch/
```

## Usage

```cpp
#include <bch/bch.hpp>
```

Create a BCH object, encode a message, and decode received codewords using the provided API.
See the `tests/` directory for examples.

The `python/` directory contains a slow pure-Python implementation intended for experiments, validation, and educational purposes.

## License

This project is licensed under the MIT License.

It was originally published under the GNU General Public License v3 (GPLv3) as part of [gr-dvbs2rx](https://github.com/igorauad/gr-dvbs2rx). This standalone version is released under MIT.

## Contributing and Support

Contributions are welcome — bug reports, improvements, and pull requests are appreciated.

If this project is useful to you, consider starring the repository or sharing it with others.

Sponsorships or other forms of support for continued development are also welcome — feel free to open an issue or reach out via GitHub.

## Citation

If you use this library in academic or technical work, you can cite it as:

```
Igor Freire, "bch: A BCH encoder/decoder library", GitHub repository, YYYY.
```

(Replace `YYYY` with the year of the version you used.)
