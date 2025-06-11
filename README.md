# Card Scanner Tool

Scans your local file system for stored credit card numbers using:

- **Presidio** for detection
- **Luhn algorithm** for validation
- **Text extraction** from files (PDFs, Word, Excel, ZIP, images, etc.)
- **Fast pre-scan** using binary regex
- **Parallel scanning** for performance

## Usage

### Simple CLI

```bash
python -m scanner.pipeline /path/to/scan
