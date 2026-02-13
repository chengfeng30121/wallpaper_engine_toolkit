#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wallpaper Engine MPKG Packer
Complete toolkit for creating Wallpaper Engine MPKG files

Features:
- Pack files and directories into MPKG format
- Support for hierarchical directory structures
- Automatic file path length validation (255 bytes limit)
- Unicode filename support
- Comprehensive packing statistics
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
class FileEntry:
    """Represents a file entry for MPKG packing"""
    file_name: str
    file_length: int
    file_start: int | None = None
    file_stop: int | None = None


def int_to_mpkg_size(n: int) -> bytes:
    """Convert integer to MPKG format file size bytes"""
    hex_str = hex(n)[2:].upper()
    
    if len(hex_str) % 2 != 0:
        hex_str = '0' + hex_str
    
    hex_str = hex_str[::-1]
    hex_bytes = binascii.unhexlify(hex_str)
    
    if len(hex_bytes) < 4:
        hex_bytes += b'\x00' * (4 - len(hex_bytes))
    elif len(hex_bytes) > 4:
        hex_bytes = hex_bytes[:4]
    
    return hex_bytes


def create_file_entry(file_name: str, file_size: int) -> bytes:
    """Create file entry data for MPKG"""
    file_name_bytes = file_name.encode('utf-8')
    file_name_len = len(file_name_bytes)
    
    # File name length byte
    name_len_byte = file_name_len.to_bytes(1, 'little')
    
    # 3 padding bytes
    padding_bytes = b'\x00\x00\x00'
    
    # File name bytes
    name_bytes = file_name_bytes
    
    # File size bytes
    size_bytes = int_to_mpkg_size(file_size)
    
    # Combine all parts
    entry_data = name_len_byte + padding_bytes + name_bytes + size_bytes
    
    return entry_data


def create_mpkg_header() -> bytes:
    """Create MPKG file header"""
    # First 4 bytes might be version or identifier
    header_part1 = b'\x00\x00\x00\x00'
    
    # Magic number
    magic = b'PKGM'
    
    # Last 8 bytes might be other metadata
    header_part2 = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    
    return header_part1 + magic + header_part2


def collect_files(input_path: str) -> List[FileEntry]:
    """Collect files to be packed"""
    files = []
    base_path = Path(input_path)
    
    if base_path.is_file():
        # Single file
        file_name = base_path.name
        # Check file name length limit (MPKG uses 1 byte length field, max 255 bytes)
        if len(file_name.encode('utf-8')) > 255:
            logger.warning(f"Warning: Filename exceeds 255 bytes limit, will be truncated: {file_name}")
            encoded_name = file_name.encode('utf-8')
            truncated_name = encoded_name[:255].decode('utf-8', errors='ignore')
            file_name = truncated_name
        file_size = base_path.stat().st_size
        files.append(FileEntry(file_name, file_size))
    elif base_path.is_dir():
        # Directory, recursively collect all files
        for file_path in base_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path
                rel_path = file_path.relative_to(base_path)
                # Convert to string, use forward slash as path separator
                file_name = str(rel_path).replace('\\', '/')
                # Check file path length limit (MPKG uses 1 byte length field, max 255 bytes)
                if len(file_name.encode('utf-8')) > 255:
                    logger.warning(f"Warning: File path exceeds 255 bytes limit, will be truncated: {file_name}")
                    encoded_path = file_name.encode('utf-8')
                    truncated_path = encoded_path[:255].decode('utf-8', errors='ignore')
                    file_name = truncated_path
                file_size = file_path.stat().st_size
                files.append(FileEntry(file_name, file_size))
    
    return files


def pack_mpkg(input_path: str, output_file: str = 'output.mpkg'):
    """Pack files or directory into MPKG format"""
    # Collect files
    logger.info(f"Collecting files: {input_path}")
    files = collect_files(input_path)
    
    if not files:
        logger.error("Error: No files found to pack")
        return
    
    logger.info(f"Found {len(files)} files")
    
    # Calculate total size of file entry data
    entries_data = b''
    for file_entry in files:
        entry_data = create_file_entry(file_entry.file_name, file_entry.file_length)
        entries_data += entry_data
    
    # Add file list end marker (filename length = 0)
    entries_data += b'\x00\x00\x00\x00'
    
    # Calculate file data start position
    header = create_mpkg_header()
    header_size = len(header)
    entries_size = len(entries_data)
    
    current_offset = header_size + entries_size
    
    # Update file start and end positions
    for file_entry in files:
        file_entry.file_start = current_offset
        file_entry.file_stop = current_offset + file_entry.file_length - 1
        current_offset += file_entry.file_length
    
    # Create output directory
    output_path = Path(output_file)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write MPKG file
    logger.info(f"Packing to: {output_file}")
    with open(output_file, 'wb') as f:
        # Write file header
        f.write(header)
        
        # Write file entries
        f.write(entries_data)
        
        # Write file data
        base_path = Path(input_path)
        for file_entry in files:
            # Build full file path
            if base_path.is_file():
                file_path = base_path
            else:
                # For truncated paths, need to rebuild the correct file path
                original_rel_path = Path(file_entry.file_name)
                file_path = base_path / original_rel_path
            
            logger.info(f"  Packing file: {file_entry.file_name} ({file_entry.file_length} bytes)")
            
            # Read and write file data
            with open(file_path, 'rb') as src_file:
                file_data = src_file.read()
                f.write(file_data)
    
    logger.info(f"Packing completed! Total packed {len(files)} files")


def main():
    """Main function with enhanced command line interface"""
    parser = argparse.ArgumentParser(
        description='Wallpaper Engine MPKG Packer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported operations:
  pack  - Create MPKG files from files or directories

Features:
  - Supports both single files and entire directories
  - Preserves directory structure in MPKG files
  - Automatic file path length validation (255 bytes limit)
  - Unicode filename support
  - Progress reporting during packing

Examples:
  %(prog)s ./assets -o package.mpkg
  %(prog)s single_file.txt -o output.mpkg
        """
    )
    
    parser.add_argument('input', help='Input file or directory to pack')
    parser.add_argument('-o', '--output', default='output.mpkg', 
                       help='Output MPKG file path (default: output.mpkg)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing output file')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Validate output
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.error(f"Output file already exists: {output_path} (use --overwrite to replace)")
        return
    
    try:
        pack_mpkg(str(input_path), str(output_path))
    except Exception as e:
        logger.error(f"Error during packing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

