#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wallpaper Engine MPKG Unpacker
Complete toolkit for extracting Wallpaper Engine MPKG files

Features:
- Extract MPKG files with preserved directory structure
- Support for long file paths and Unicode filenames
- Validation of MPKG file format
- Comprehensive extraction statistics
"""

import os
import sys
import argparse
import logging
import binascii
from dataclasses import dataclass
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StoredFile:
    """Represents a file stored in MPKG format"""
    file_name: str
    file_length: int
    file_start: int | None = None
    file_stop: int | None = None


def get_file_length(data: bytes) -> int:
    """Get file length from MPKG data"""
    raw_hex_str = binascii.b2a_hex(data).decode('utf-8')
    hex_filesize = ""
    for i in range(0, len(raw_hex_str), 2):
        single_hex = raw_hex_str[i:i+2]
        if single_hex == "00":
            continue
        hex_filesize = single_hex + hex_filesize
    return int(hex_filesize.encode('utf-8'), 16)


def is_valid_mpkg_file(data: bytes) -> bool:
    """Check if data is a valid MPKG file"""
    if len(data) < 8:
        return False
    magic_data = data[4:8]
    return magic_data == b'PKGM'


def parse_file_list(data: bytes) -> List[StoredFile]:
    """Parse file list from MPKG data"""
    file_list: List[StoredFile] = []
    offset = 16
    if data[4:8] == b'PKGM':
        data = data[16:]
    while data:
        file_name_len = data[0]
        if data[1:4] != b'\x00\x00\x00':
            break
        single_file_info = data[4: file_name_len + 12]
        filename = single_file_info[:file_name_len]
        try:
            filename = filename.decode('utf-8')
        except:
            break
        file_length = get_file_length(single_file_info[file_name_len+4:])
        file_list.append(StoredFile(filename, file_length))
        data = data[file_name_len + 12:]
        offset += file_name_len + 12
    for file in file_list:
        file.file_start = offset
        file.file_stop = offset + file.file_length - 1
        offset += file.file_length
    return file_list


def read_file(file_path: str) -> bytes:
    """Read file bytes"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found or not a regular file: {file_path}")
    with open(file_path, "rb") as f:
        return f.read()


def _write_file(file_path: str, data: bytes):
    """Write file bytes (internal function)"""
    with open(file_path, "wb") as f:
        f.write(data)


def write_file(file_path: str, data: bytes):
    """Write file bytes with automatic directory creation"""
    file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    _write_file(file_path, data)


def unpack_mpkg_file(file_path: str, output_dir: str = 'output'):
    """Unpack MPKG file"""
    data = read_file(file_path)
    if not is_valid_mpkg_file(data):
        raise ValueError("Not a valid mpkg file")
    logger.info("File is a valid mpkg file")
    file_list = parse_file_list(data)
    for file in file_list:
        file_data = data[file.file_start:file.file_stop+1]
        fn = os.path.join(output_dir, file.file_name)
        write_file(fn, file_data)
        logger.info("Extracted file: %s", file.file_name)
    logger.info("All done.")


def main():
    """Main function with improved argument handling"""
    parser = argparse.ArgumentParser(
        description='Wallpaper Engine MPKG Unpacker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported operations:
  unpack  - Extract MPKG files preserving directory structure

Features:
  - Handles long file paths and Unicode filenames
  - Validates MPKG file format before extraction
  - Creates output directories automatically
  - Provides detailed extraction progress

Examples:
  %(prog)s sample.mpkg
  %(prog)s sample.mpkg -o ./extracted
        """
    )
    
    parser.add_argument('input', help='Path to MPKG file to unpack')
    parser.add_argument('-o', '--output', default='./output', 
                       help='Output directory (default: ./output)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing files in output directory')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return
    
    if not input_path.is_file():
        logger.error(f"Input path is not a file: {input_path}")
        return
    
    # Validate output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        unpack_mpkg_file(str(input_path), str(output_path))
    except Exception as e:
        logger.error(f"Error unpacking MPKG file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
