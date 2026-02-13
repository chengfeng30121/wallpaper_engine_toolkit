# Wallpaper Engine Toolkit

A comprehensive Python toolkit for working with Wallpaper Engine PKG and MPKG files. This toolkit provides utilities for extracting, packing, converting, and analyzing Wallpaper Engine asset files.

## Features

- **PKG/MPKG Extraction**: Extract files from both PKG and MPKG archive formats
- **TEX Conversion**: Convert TEX texture files to standard image formats (PNG, GIF, MP4)
- **Archive Creation**: Pack files and directories into PKG or MPKG formats
- **Format Conversion**: Convert between PKG and MPKG formats
- **Animated GIF Support**: Handle animated TEX files with frame information
- **Video Texture Support**: Extract MP4 videos from TEX files
- **Comprehensive CLI**: Unified command-line interface with extensive options

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  ```bash
  pip install Pillow lz4
  ```

### Optional Dependencies

- **Pillow**: Required for image conversion and TEX processing
- **lz4**: Required for TEX compression/decompression

## Quick Start

### Extract PKG/MPKG Files

```bash
# Extract a PKG file
python wallpaper_engine_toolkit.py extract package.pkg -o ./output

# Extract an MPKG file
python wallpaper_engine_toolkit.py extract package.mpkg --mpkg -o ./output

# Extract with automatic TEX conversion
python wallpaper_engine_toolkit.py extract package.pkg -o ./output --overwrite
```

### Convert TEX Files

```bash
# Convert single TEX file
python wallpaper_engine_toolkit.py convert image.tex

# Convert all TEX files in directory
python wallpaper_engine_toolkit.py convert ./textures -r --overwrite
```

### Pack Files

```bash
# Pack directory into PKG format
python wallpaper_engine_toolkit.py pack ./assets -o package.pkg --pkg

# Pack directory into MPKG format
python wallpaper_engine_toolkit.py pack ./assets -o package.mpkg --mpkg
```

### Format Conversion

```bash
# Convert PKG to MPKG
python wallpaper_engine_toolkit.py convert-format package.pkg -o package.mpkg --to-mpkg

# Convert MPKG to PKG
python wallpaper_engine_toolkit.py convert-format package.mpkg -o package.pkg --to-pkg
```

## Detailed Usage

### Main Toolkit (wallpaper_engine_toolkit.py)

The unified toolkit supports all operations:

```bash
python wallpaper_engine_toolkit.py [command] [options]
```

#### Commands

- `extract` - Extract PKG/MPKG files and convert TEX to images
- `pack` - Pack files into PKG/MPKG format
- `convert` - Convert TEX files to images
- `convert-format` - Convert between PKG and MPKG formats
- `info` - Display file information

#### Common Options

- `-o, --output PATH` - Output directory or file path
- `--overwrite` - Overwrite existing files
- `--pkg` - Force PKG format (highest priority)
- `--mpkg` - Force MPKG format (highest priority)

#### Extract Options

- `-i, --ignoreexts EXT` - Ignore files with specified extensions (comma separated)
- `-e, --onlyexts EXT` - Only extract files with specified extensions (comma separated)
- `-s, --singledir` - Put all extracted files in one directory
- `-r, --recursive` - Recursive search in subdirectories
- `--no-tex-convert` - Do not convert TEX files to images

### Individual Tools

The toolkit also provides standalone tools for specific operations:

#### PKG Unpacker (pkg_unpacker.py)
```bash
python pkg_unpacker.py extract package.pkg -o ./output
```

#### PKG Packer (pkg_packer.py)
```bash
python pkg_packer.py ./assets -o package.pkg
```

#### MPKG Unpacker (mpkg_unpacker.py)
```bash
python mpkg_unpacker.py sample.mpkg -o ./extracted
```

#### MPKG Packer (mpkg_packer.py)
```bash
python mpkg_packer.py ./assets -o package.mpkg
```

## File Format Support

### PKG Format
- Supports long file paths (length-prefixed strings)
- Entry types: JSON, TEX, shaders (FRAG/VERT), audio (MP3), images (JPG/GIF)
- Automatic TEX generation from standard image files

### MPKG Format
- 255-byte filename limit (1-byte length field)
- Hierarchical directory structure preservation
- Unicode filename support

### TEX Format
- Multiple texture formats: RGBA8888, DXT1, DXT3, DXT5, R8, RG88
- LZ4 compression support
- Animated GIF texture support
- Video texture extraction (MP4)
- Mipmap support with various output formats

## Advanced Features

### Experimental Long String Support

Enable experimental PKG-style parsing for MPKG files:

```bash
python wallpaper_engine_toolkit.py extract sample.mpkg --long-string-support -o ./output
```

### File Filtering

Extract only specific file types:

```bash
# Extract only JSON and shader files
python wallpaper_engine_toolkit.py extract package.pkg -e .json,.frag,.vert -o ./filtered

# Skip audio files
python wallpaper_engine_toolkit.py extract package.pkg -i .mp3,.wav -o ./no_audio
```

### Batch Processing

Process multiple files or directories:

```bash
# Process all PKG files in directory
python wallpaper_engine_toolkit.py extract ./packages -r -o ./extracted

# Convert all TEX files recursively
python wallpaper_engine_toolkit.py convert ./textures -r --overwrite
```

## Technical Details

### TEX Conversion Process

1. **Header Parsing**: Reads TEXV0005 and TEXI0001 magic strings
2. **Format Detection**: Identifies texture format (RGBA8888, DXT*, etc.)
3. **Container Analysis**: Processes TEXB containers (versions 0001-0004)
4. **Mipmap Extraction**: Reads compressed/uncompressed mipmap data
5. **Decompression**: Handles LZ4 compression when present
6. **Format Conversion**: Converts to standard image formats
7. **Animation Handling**: Processes frame information for GIF textures

### Format Detection Priority

1. `--pkg` / `--mpkg` flags (highest priority)
2. File extension (.pkg/.mpkg)
3. File content analysis
4. Default to PKG format

## Error Handling

The toolkit provides comprehensive error handling:

- **File Validation**: Checks file format validity before processing
- **Dependency Checking**: Verifies required libraries (PIL, lz4)
- **Graceful Degradation**: Continues processing when possible despite errors
- **Detailed Logging**: Provides informative error messages and progress updates

## Performance Considerations

- **Memory Usage**: Large archives are processed in chunks to minimize memory footprint
- **Compression**: LZ4 compression provides good balance of speed and size reduction
- **Parallel Processing**: Consider processing multiple files in batches for better performance

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install Pillow lz4
   ```

2. **Permission Errors**: Ensure write permissions for output directories

3. **Large File Processing**: For very large archives, consider increasing system memory or processing in smaller batches

4. **Encoding Issues**: The toolkit handles Unicode filenames properly, but ensure your system locale supports UTF-8

### Debug Mode

Add `--debug` flag for detailed error information:
```bash
python wallpaper_engine_toolkit.py extract package.pkg --debug
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate documentation
- Tests cover new functionality
- Pull requests include clear descriptions of changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wallpaper Engine by Kristjan Skutta
- LZ4 compression library
- Python Imaging Library (Pillow)
- Community contributors and testers

---

*Note: This toolkit is not officially affiliated with Wallpaper Engine or Valve Corporation. Use responsibly and respect copyright laws.*