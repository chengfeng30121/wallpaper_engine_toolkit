#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wallpaper Engine PKG Packer
Complete toolkit for creating Wallpaper Engine PKG files

Features:
- Create PKG files from directory structures
- Automatic TEX file generation from images
- Support for various file types (JSON, TEX, shaders, audio, etc.)
- LZ4 compression for TEX mipmaps
- Comprehensive packing statistics
"""
import os
import sys
import struct
import argparse
import logging
from pathlib import Path
from typing import List, BinaryIO
from enum import IntEnum
from dataclasses import dataclass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not installed. Image conversion features will be disabled.")

try:
    import lz4.block
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    print("Warning: lz4 not installed. Some TEX compression features may not work.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PackageEntry:
    """Represents a single entry in a PKG file"""
    name: str
    extension: str
    directory_path: str
    full_path: str
    type: int  # EntryType
    bytes: bytes
    
    @property
    def full_filename(self) -> str:
        """Get full filename with extension"""
        return self.name + self.extension


class EntryType(IntEnum):
    """Entry types in PKG files"""
    UNKNOWN = 0
    JSON = 1
    TEX = 2
    FRAG = 3
    VERT = 4
    MP3 = 5
    JPG = 6
    GIF = 7
    
    @classmethod
    def from_extension(cls, ext: str) -> 'EntryType':
        """Determine entry type from file extension"""
        ext_lower = ext.lower()
        if ext_lower == '.json':
            return cls.JSON
        elif ext_lower in ['.tex', '.png', '.jpg', '.jpeg']:
            return cls.TEX
        elif ext_lower == '.frag':
            return cls.FRAG
        elif ext_lower == '.vert':
            return cls.VERT
        elif ext_lower == '.mp3':
            return cls.MP3
        elif ext_lower in ['.jpg', '.jpeg']:
            return cls.JPG
        elif ext_lower == '.gif':
            return cls.GIF
        else:
            return cls.UNKNOWN


class TexFormat(IntEnum):
    """TEX texture formats"""
    RGBA8888 = 0
    DXT1 = 1
    DXT3 = 2
    DXT5 = 3
    R8 = 4
    RG88 = 5
    
    @classmethod
    def from_string(cls, format_str: str) -> 'TexFormat':
        """Convert string to TexFormat"""
        format_map = {
            'rgba8888': cls.RGBA8888,
            'dxt1': cls.DXT1,
            'dxt3': cls.DXT3,
            'dxt5': cls.DXT5,
            'r8': cls.R8,
            'rg88': cls.RG88
        }
        return format_map.get(format_str.lower(), cls.RGBA8888)


class MipmapFormat(IntEnum):
    """Mipmap formats for output"""
    # FreeImageFormat values (0-35)
    IMAGE_BMP = 0
    IMAGE_ICO = 1  
    IMAGE_JPEG = 2
    IMAGE_JNG = 3
    IMAGE_KOALA = 4
    IMAGE_LBM = 5
    IMAGE_MNG = 6
    IMAGE_PBM = 7
    IMAGE_PBMRAW = 8
    IMAGE_PCD = 9
    IMAGE_PCX = 10
    IMAGE_PGM = 11
    IMAGE_PGMRAW = 12
    IMAGE_PNG = 13
    IMAGE_PPM = 14
    IMAGE_PPMRAW = 15
    IMAGE_RAS = 16
    IMAGE_TARGA = 17
    IMAGE_TIFF = 18
    IMAGE_WBMP = 19
    IMAGE_PSD = 20
    IMAGE_CUT = 21
    IMAGE_XBM = 22
    IMAGE_XPM = 23
    IMAGE_DDS = 24
    IMAGE_GIF = 25
    IMAGE_HDR = 26
    IMAGE_FAXG3 = 27
    IMAGE_SGI = 28
    IMAGE_EXR = 29
    IMAGE_J2K = 30
    IMAGE_JP2 = 31
    IMAGE_PFM = 32
    IMAGE_PICT = 33
    IMAGE_RAW = 34
    VIDEO_MP4 = 35
    
    # Raw and compressed formats (negative values or custom)
    R8 = -100
    RG88 = -101
    RGBA8888 = -102
    COMPRESSED_DXT1 = -103
    COMPRESSED_DXT3 = -104  
    COMPRESSED_DXT5 = -105
    
    @classmethod
    def from_extension(cls, ext: str) -> 'MipmapFormat':
        """Get mipmap format from file extension"""
        ext_map = {
            '.png': cls.IMAGE_PNG,
            '.jpg': cls.IMAGE_JPEG,
            '.jpeg': cls.IMAGE_JPEG,
            '.bmp': cls.IMAGE_BMP,
            '.gif': cls.IMAGE_GIF,
            '.mp4': cls.VIDEO_MP4
        }
        return ext_map.get(ext.lower(), cls.IMAGE_PNG)


@dataclass
class PackingStats:
    """Statistics for packing process"""
    total: int = 0
    success: int = 0
    failed: int = 0
    
    def add_total(self):
        self.total += 1
    
    def add_success(self):
        self.success += 1
    
    def add_failed(self):
        self.failed += 1
    
    def print_summary(self):
        logger.info(f"\n### Packing Summary ###")
        logger.info(f"Total files: {self.total}")
        logger.info(f"Successfully processed: {self.success}")
        logger.info(f"Failed: {self.failed}")


def determine_entry_type(file_path: str) -> int:
    """Determine the entry type based on file extension"""
    path_obj = Path(file_path)
    return EntryType.from_extension(path_obj.suffix).value


def convert_image_to_tex_format(image_path: str, target_format: TexFormat) -> bytes:
    """Convert image file to TEX raw data in specified format"""
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow is required for image conversion")
    
    try:
        with Image.open(image_path) as img:
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            width, height = img.size
            rgba_data = img.tobytes()
            
            if target_format == TexFormat.RGBA8888:
                # Already in correct format
                return rgba_data
            
            elif target_format == TexFormat.R8:
                # Extract red channel only
                r_data = rgba_data[0::4]
                return r_data
            
            elif target_format == TexFormat.RG88:
                # Extract red and green channels
                rg_data = b''.join([rgba_data[i:i+2] for i in range(0, len(rgba_data), 4)])
                return rg_data
            
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
    except Exception as e:
        logger.error(f"Failed to convert image {image_path} to TEX format: {e}")
        raise


def create_tex_header(format_val: int, is_gif: bool = False, flags: int = 0) -> bytes:
    """Create TEX header bytes"""
    header = bytearray()
    
    # Magic strings (null-terminated, 16 bytes each)
    texv_magic = b"TEXV0005\x00\x00\x00\x00\x00\x00\x00"
    texi_magic = b"TEXI0001\x00\x00\x00\x00\x00\x00\x00"
    
    header.extend(texv_magic)
    header.extend(texi_magic)
    
    # Header fields
    header.extend(struct.pack('<i', format_val))           # format
    header.extend(struct.pack('<i', flags))                # flags
    header.extend(struct.pack('<i', 0))                    # texture_width
    header.extend(struct.pack('<i', 0))                    # texture_height
    header.extend(struct.pack('<i', 0))                    # image_width
    header.extend(struct.pack('<i', 0))                    # image_height
    header.extend(struct.pack('<I', 0))                    # unk_int0 (unsigned)
    
    return bytes(header)


def create_tex_image_container(magic: str, image_count: int, image_format: int = -1) -> bytes:
    """Create TEX image container"""
    container = bytearray()
    
    # Magic string (null-terminated, 16 bytes)
    magic_bytes = magic.encode('ascii') + b'\x00' * (16 - len(magic))
    container.extend(magic_bytes)
    
    # Image count
    container.extend(struct.pack('<i', image_count))
    
    # For TEXB0003 and TEXB0004, add image format
    if magic in ['TEXB0003', 'TEXB0004']:
        container.extend(struct.pack('<i', image_format))
        
    # For TEXB0004, add is_video_mp4 flag
    if magic == 'TEXB0004':
        container.extend(struct.pack('<i', 0))  # is_video_mp4 = false
        
    return bytes(container)


def compress_with_lz4(data: bytes) -> bytes:
    """Compress data with LZ4"""
    if not LZ4_AVAILABLE:
        raise RuntimeError("lz4 library is required for compression")
    return lz4.block.compress(data, store_size=False)


def create_mipmap_data(width: int, height: int, raw_data: bytes, use_lz4: bool = True) -> bytes:
    """Create mipmap data with optional LZ4 compression"""
    mipmap = bytearray()
    
    # Width and height
    mipmap.extend(struct.pack('<i', width))
    mipmap.extend(struct.pack('<i', height))
    
    if use_lz4 and LZ4_AVAILABLE:
        # Compressed format (similar to TEXB0003/0004)
        is_lz4 = 1
        decompressed_size = len(raw_data)
        compressed_data = compress_with_lz4(raw_data)
        compressed_size = len(compressed_data)
        
        mipmap.extend(struct.pack('<i', is_lz4))
        mipmap.extend(struct.pack('<i', decompressed_size))
        mipmap.extend(struct.pack('<i', compressed_size))
        mipmap.extend(compressed_data)
    else:
        # Uncompressed format (similar to TEXB0001)
        data_size = len(raw_data)
        mipmap.extend(struct.pack('<i', data_size))
        mipmap.extend(raw_data)
    
    return bytes(mipmap)


def convert_image_to_tex_file(image_path: str, tex_format: TexFormat = TexFormat.RGBA8888) -> bytes:
    """Convert image file to complete TEX file bytes"""
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow is required for image conversion")
    
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            width, height = img.size
        
        # Get raw data in target format
        raw_data = convert_image_to_tex_format(image_path, tex_format)
        
        # Create TEX header
        tex_header = create_tex_header(tex_format.value)
        
        # Create image container (use TEXB0003 for compressed format)
        image_container = create_tex_image_container('TEXB0003', 1, -1)  # FIF_UNKNOWN
        
        # Create mipmap data (with LZ4 compression)
        mipmap_data = create_mipmap_data(width, height, raw_data, use_lz4=LZ4_AVAILABLE)
        
        # Combine all parts
        tex_file = bytearray()
        tex_file.extend(tex_header)
        tex_file.extend(image_container)
        tex_file.extend(mipmap_data)
        
        return bytes(tex_file)
        
    except Exception as e:
        logger.error(f"Failed to create TEX file from {image_path}: {e}")
        raise


def scan_directory_for_pkg_entries(input_dir: str) -> List[PackageEntry]:
    """Scan directory and create package entries"""
    entries = []
    input_path = Path(input_dir)
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file():
            # Skip .tex-json files and other metadata
            if file_path.suffix == '.tex-json':
                continue
            
            # Check file path length limitation (255 bytes limit for PKG format)
            relative_path = file_path.relative_to(input_path)
            full_path_str = str(relative_path).replace('\\', '/')
            if len(full_path_str.encode('utf-8')) > 255:
                logger.warning(f"Warning: File path exceeds 255 bytes limit, will be truncated: {full_path_str}")
                encoded_path = full_path_str.encode('utf-8')
                truncated_path = encoded_path[:255].decode('utf-8', errors='ignore')
                full_path_str = truncated_path
            
            # Split path into directory and filename
            if '/' in full_path_str:
                directory_path = '/'.join(full_path_str.split('/')[:-1])
                filename = full_path_str.split('/')[-1]
            else:
                directory_path = ''
                filename = full_path_str
            
            # Split filename into name and extension
            if '.' in filename:
                name = '.'.join(filename.split('.')[:-1])
                extension = '.' + filename.split('.')[-1]
            else:
                name = filename
                extension = ''
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Determine entry type
            entry_type = determine_entry_type(str(file_path))
            
            # Handle image files that need to be converted to TEX
            if entry_type == EntryType.TEX and (file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']):
                try:
                    # Check if there's a corresponding .tex file
                    tex_path = file_path.with_suffix('.tex')
                    if tex_path.exists():
                        # Use existing TEX file
                        with open(tex_path, 'rb') as f:
                            file_bytes = f.read()
                    else:
                        # Convert image to TEX
                        logger.info(f"* Converting {file_path} to TEX format")
                        file_bytes = convert_image_to_tex_file(str(file_path))
                        extension = '.tex'
                        entry_type = EntryType.TEX
                        
                except Exception as e:
                    logger.warning(f"Failed to convert {file_path} to TEX, using original: {e}")
            
            entry = PackageEntry(
                name=name,
                extension=extension,
                directory_path=directory_path,
                full_path=full_path_str,
                type=entry_type,
                bytes=file_bytes
            )
            entries.append(entry)
    
    return entries


def write_string_i32_size(writer: BinaryIO, input_str: str):
    """Write string with 32-bit length prefix"""
    data = input_str.encode('utf-8')
    writer.write(struct.pack('<i', len(data)))
    writer.write(data)


def write_pkg_file(entries: List[PackageEntry], output_path: str):
    """Write entries to PKG file"""
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Write magic (length-prefixed string)
        write_string_i32_size(f, "PKGV0002")
        
        # Write entry count
        f.write(struct.pack('<i', len(entries)))
        
        # Write entry metadata
        current_offset = 0
        for entry in entries:
            # Write full path (length-prefixed string)
            write_string_i32_size(f, entry.full_path)
            
            # Write offset and size
            f.write(struct.pack('<i', current_offset))
            f.write(struct.pack('<i', len(entry.bytes)))
            
            current_offset += len(entry.bytes)
        
        # Write entry data
        for entry in entries:
            f.write(entry.bytes)


def create_pkg_from_directory(input_dir: str, output_file: str, stats: PackingStats):
    """Create PKG file from directory"""
    logger.info(f"\n### Creating PKG from directory: {input_dir}")
    
    try:
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        # Scan directory for entries
        entries = scan_directory_for_pkg_entries(input_dir)
        
        if not entries:
            logger.warning("No files found in directory")
            return
        
        # Update statistics
        stats.total += len(entries)
        
        # Write PKG file
        write_pkg_file(entries, output_file)
        stats.success += len(entries)
        
        logger.info(f"* Created PKG file: {output_file} ({len(entries)} entries)")
        
    except Exception as e:
        logger.error(f"Failed to create PKG from {input_dir}: {e}")
        stats.failed += 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Wallpaper Engine PKG Packer - Create PKG files from directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported operations:
  pack  - Create PKG file from directory structure

Features:
  - Automatic TEX file generation from images (PNG/JPG)
  - LZ4 compression for optimal file sizes
  - Support for various file types (JSON, shaders, audio, etc.)
  - File path length validation (255 bytes limit)

Examples:
  %(prog)s ./assets -o package.pkg
  %(prog)s ./textures -o textures.pkg --overwrite
        """
    )
    
    parser.add_argument('input', help='Input directory containing files to pack')
    parser.add_argument('-o', '--output', required=True, help='Output PKG file path')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file')
    parser.add_argument('--tex-format', choices=['rgba8888', 'r8', 'rg88'], 
                       default='rgba8888', help='TEX format for image conversion (default: rgba8888)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        return
    
    # Validate output
    if output_path.exists() and not args.overwrite:
        logger.error(f"Output file already exists: {output_path} (use --overwrite to replace)")
        return
    
    # Validate dependencies
    if not PIL_AVAILABLE:
        logger.error("Error: PIL/Pillow is required for image operations")
        logger.error("Install with: pip install Pillow")
        return
    
    if not LZ4_AVAILABLE:
        logger.warning("Warning: lz4 library not available. TEX files will not be compressed.")
        logger.warning("Install with: pip install lz4")
    
    # Initialize statistics
    stats = PackingStats()
    
    try:
        create_pkg_from_directory(str(input_path), str(output_path), stats)
        stats.print_summary()
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        stats.print_summary()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        stats.print_summary()


if __name__ == '__main__':
    main()