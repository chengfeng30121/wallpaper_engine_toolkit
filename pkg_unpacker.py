#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wallpaper Engine PKG Unpacker
Complete toolkit for extracting and converting Wallpaper Engine PKG files

Features:
- Extract PKG files with full directory structure preservation
- Convert TEX files to PNG/GIF/MP4 formats
- Support for animated GIF textures
- Video texture extraction (MP4)
- Comprehensive file filtering options
- Detailed extraction statistics
"""
import os
import sys
import struct
import json
import argparse
import logging
import binascii
from pathlib import Path
from typing import List, Optional, Tuple, BinaryIO
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


@dataclass
class ExtractionStats:
    """Statistics for extraction process"""
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
        logger.info(f"\n### Extraction Summary ###")
        logger.info(f"Total files: {self.total}")
        logger.info(f"Successfully processed: {self.success}")
        logger.info(f"Failed: {self.failed}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EntryType(IntEnum):
    """Package entry types"""
    BINARY = 0
    TEX = 1


class TexFormat(IntEnum):
    """TEX texture formats"""
    RGBA8888 = 0
    DXT5 = 4
    DXT3 = 6
    DXT1 = 7
    RG88 = 8
    R8 = 9


class TexFlags(IntEnum):
    """TEX flags (bitwise)"""
    NONE = 0
    NO_INTERPOLATION = 1
    CLAMP_UVS = 2
    IS_GIF = 4
    UNK3 = 8
    UNK4 = 16
    IS_VIDEO_TEXTURE = 32
    UNK6 = 64
    UNK7 = 128


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


def get_entry_type_from_filename(path: str) -> EntryType:
    """Get entry type from file extension"""
    ext = Path(path).suffix.lower()
    if ext == '.tex':
        return EntryType.TEX
    return EntryType.BINARY


def read_string_i32_size(reader: BinaryIO, max_length: int = -1) -> str:
    """Read string with 32-bit length prefix"""
    size_data = reader.read(4)
    if len(size_data) < 4:
        raise ValueError("Unexpected end of file while reading string size")
    
    size = struct.unpack('<i', size_data)[0]
    if size < 0:
        raise ValueError(f"String size cannot be negative: {size}")
    
    if max_length > -1:
        size = min(size, max_length)
    
    if size == 0:
        return ""
    
    data = reader.read(size)
    if len(data) < size:
        raise ValueError(f"Unexpected end of file while reading string data (expected {size}, got {len(data)})")
    
    return data.decode('utf-8')


def write_string_i32_size(writer: BinaryIO, input_str: str):
    """Write string with 32-bit length prefix"""
    data = input_str.encode('utf-8')
    writer.write(struct.pack('<i', len(data)))
    writer.write(data)


@dataclass
class PackageEntry:
    """Represents a single entry in a PKG file"""
    full_path: str = ""
    offset: int = 0
    length: int = 0
    bytes: bytes = b""
    type: EntryType = EntryType.BINARY
    
    @property
    def name(self) -> str:
        return Path(self.full_path).stem
    
    @property
    def extension(self) -> str:
        return Path(self.full_path).suffix
    
    @property
    def directory_path(self) -> str:
        return str(Path(self.full_path).parent)


@dataclass
class Package:
    """Represents a PKG file"""
    magic: str = ""
    header_size: int = 0
    entries: List[PackageEntry] = None
    
    def __post_init__(self):
        if self.entries is None:
            self.entries = []


@dataclass
class TexHeader:
    """TEX header structure"""
    format: TexFormat = TexFormat.RGBA8888
    flags: TexFlags = TexFlags.NONE
    texture_width: int = 0
    texture_height: int = 0
    image_width: int = 0
    image_height: int = 0
    unk_int0: int = 0
    
    def has_flag(self, flag: TexFlags) -> bool:
        """Check if header has specific flag"""
        return bool(self.flags & flag)
    
    @property
    def is_gif(self) -> bool:
        return self.has_flag(TexFlags.IS_GIF)
    
    @property
    def is_video_texture(self) -> bool:
        return self.has_flag(TexFlags.IS_VIDEO_TEXTURE)


@dataclass
class TexMipmap:
    """TEX mipmap data"""
    width: int = 0
    height: int = 0
    format: TexFormat = TexFormat.RGBA8888
    bytes: bytes = b""


@dataclass
class TexImage:
    """TEX image containing mipmaps"""
    mipmaps: List[TexMipmap] = None
    
    def __post_init__(self):
        if self.mipmaps is None:
            self.mipmaps = []
    
    @property
    def first_mipmap(self) -> Optional[TexMipmap]:
        return self.mipmaps[0] if self.mipmaps else None


@dataclass
class TexFrameInfo:
    """TEX frame information for GIF animations"""
    image_id: int = 0
    frametime: float = 0.0
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    width_y: float = 0.0
    height_x: float = 0.0
    height: float = 0.0


@dataclass
class TexFrameInfoContainer:
    """Container for TEX frame information"""
    magic: str = ""
    frames: List[TexFrameInfo] = None
    gif_width: int = 0
    gif_height: int = 0
    
    def __post_init__(self):
        if self.frames is None:
            self.frames = []


class PackageReader:
    """Reader for PKG files"""
    
    def __init__(self, read_entry_bytes: bool = True):
        self.read_entry_bytes = read_entry_bytes
    
    def read_from(self, reader: BinaryIO) -> Package:
        """Read PKG from binary reader"""
        package_start = reader.tell()
        package = Package()
        package.magic = read_string_i32_size(reader, 32)
        
        self._read_entries(package.entries, reader)
        
        data_start = reader.tell()
        package.header_size = data_start - package_start
        
        if self.read_entry_bytes:
            self._populate_entries_with_data(data_start, package.entries, reader)
        
        return package
    
    def _read_entries(self, entries: List[PackageEntry], reader: BinaryIO):
        """Read package entries header"""
        entry_count_data = reader.read(4)
        if len(entry_count_data) < 4:
            raise ValueError("Unexpected end of file while reading entry count")
        
        entry_count = struct.unpack('<i', entry_count_data)[0]
        
        for i in range(entry_count):
            full_path = read_string_i32_size(reader, 255)
            offset_data = reader.read(4)
            length_data = reader.read(4)
            
            if len(offset_data) < 4 or len(length_data) < 4:
                raise ValueError(f"Unexpected end of file while reading entry {i}")
            
            offset = struct.unpack('<i', offset_data)[0]
            length = struct.unpack('<i', length_data)[0]
            
            entry_type = get_entry_type_from_filename(full_path)
            entries.append(PackageEntry(full_path, offset, length, b"", entry_type))
    
    def _populate_entries_with_data(self, data_start: int, entries: List[PackageEntry], reader: BinaryIO):
        """Populate entries with actual data bytes"""
        for entry in entries:
            reader.seek(entry.offset + data_start)
            entry.bytes = reader.read(entry.length)


class DXTFlags(IntEnum):
    """DXT compression flags"""
    DXT1 = 1
    DXT3 = 2
    DXT5 = 4


def decompress_dxt1_color_block(block: bytes, is_dxt1: bool = True) -> List[int]:
    """Decompress DXT1 color block"""
    # Unpack endpoints
    color0 = struct.unpack('<H', block[0:2])[0]
    color1 = struct.unpack('<H', block[2:4])[0]
    
    # Decode RGB565
    def rgb565_to_rgb888(color):
        r = ((color >> 11) & 0x1F) << 3
        g = ((color >> 5) & 0x3F) << 2
        b = (color & 0x1F) << 3
        return r | (r >> 5), g | (g >> 6), b | (b >> 5)
    
    r0, g0, b0 = rgb565_to_rgb888(color0)
    r1, g1, b1 = rgb565_to_rgb888(color1)
    
    # Generate palette
    colors = [
        [r0, g0, b0, 255],
        [r1, g1, b1, 255]
    ]
    
    if is_dxt1 and color0 <= color1:
        # 3-color mode + transparent
        colors.append([(r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255])
        colors.append([0, 0, 0, 0])
    else:
        # 4-color mode
        colors.append([(2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3, 255])
        colors.append([(r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3, 255])
    
    # Read indices
    indices = 0
    for i in range(4):
        indices |= block[4 + i] << (8 * i)
    
    # Extract pixels
    pixels = []
    for i in range(16):
        index = (indices >> (2 * i)) & 0x3
        pixels.extend(colors[index])
    
    return pixels


def decompress_dxt_alpha_dxt3(alpha_block: bytes) -> List[int]:
    """Decompress DXT3 alpha"""
    alpha = []
    for i in range(8):
        quant = alpha_block[i]
        lo = quant & 0x0F
        hi = (quant & 0xF0) >> 4
        alpha.append((lo << 4) | lo)  # Expand 4-bit to 8-bit
        alpha.append((hi << 4) | hi)
    return alpha


def decompress_dxt_alpha_dxt5(alpha_block: bytes) -> List[int]:
    """Decompress DXT5 alpha"""
    alpha0 = alpha_block[0]
    alpha1 = alpha_block[1]
    
    # Build codebook
    codes = [alpha0, alpha1]
    if alpha0 <= alpha1:
        # 5-alpha codebook + 0 + 255
        for i in range(1, 5):
            codes.append(((5 - i) * alpha0 + i * alpha1) // 5)
        codes.extend([0, 255])
    else:
        # 7-alpha codebook
        for i in range(1, 7):
            codes.append(((7 - i) * alpha0 + i * alpha1) // 7)
    
    # Decode indices
    indices_data = 0
    for i in range(6):
        indices_data |= alpha_block[2 + i] << (8 * i)
    
    indices = []
    for i in range(16):
        indices.append((indices_data >> (3 * i)) & 0x7)
    
    # Apply codebook
    alpha = [codes[idx] for idx in indices]
    return alpha


def decompress_dxt_image(width: int, height: int, data: bytes, dxt_flags: DXTFlags) -> bytes:
    """Decompress DXT image"""
    rgba = bytearray(width * height * 4)
    
    bytes_per_block = 8 if dxt_flags == DXTFlags.DXT1 else 16
    block_index = 0
    
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if block_index >= len(data):
                break
            
            # Decompress color
            if dxt_flags == DXTFlags.DXT1:
                color_data = data[block_index:block_index + 8]
                pixels = decompress_dxt1_color_block(color_data, True)
                alpha_data = [255] * 16
            elif dxt_flags in (DXTFlags.DXT3, DXTFlags.DXT5):
                alpha_block = data[block_index:block_index + 8]
                color_block = data[block_index + 8:block_index + 16]
                pixels = decompress_dxt1_color_block(color_block, False)
                
                if dxt_flags == DXTFlags.DXT3:
                    alpha_data = decompress_dxt_alpha_dxt3(alpha_block)
                else:  # DXT5
                    alpha_data = decompress_dxt_alpha_dxt5(alpha_block)
            else:
                raise ValueError(f"Unsupported DXT flags: {dxt_flags}")
            
            # Write pixels to output
            pixel_idx = 0
            for py in range(4):
                for px in range(4):
                    sx, sy = x + px, y + py
                    if sx < width and sy < height:
                        dst_idx = 4 * (sy * width + sx)
                        rgba[dst_idx + 0] = pixels[pixel_idx * 4 + 0]     # R
                        rgba[dst_idx + 1] = pixels[pixel_idx * 4 + 1]     # G
                        rgba[dst_idx + 2] = pixels[pixel_idx * 4 + 2]     # B
                        rgba[dst_idx + 3] = alpha_data[pixel_idx]         # A
                    pixel_idx += 1
            
            block_index += bytes_per_block
    
    return bytes(rgba)


def convert_rg88_to_rgba8888(data: bytes, width: int, height: int) -> bytes:
    """Convert RG88 format to RGBA8888"""
    if len(data) != width * height * 2:
        raise ValueError(f"RG88 data size mismatch: expected {width * height * 2}, got {len(data)}")
    
    rgba = bytearray(width * height * 4)
    for i in range(width * height):
        r = data[i * 2]      # Red channel
        g = data[i * 2 + 1]  # Green channel (used as grayscale)
        rgba[i * 4 + 0] = g  # R = G (grayscale)
        rgba[i * 4 + 1] = g  # G = G
        rgba[i * 4 + 2] = g  # B = G
        rgba[i * 4 + 3] = r  # A = R
    
    return bytes(rgba)


def convert_r8_to_rgba8888(data: bytes, width: int, height: int) -> bytes:
    """Convert R8 format to RGBA8888 (grayscale)"""
    if len(data) != width * height:
        raise ValueError(f"R8 data size mismatch: expected {width * height}, got {len(data)}")
    
    rgba = bytearray(width * height * 4)
    for i in range(width * height):
        gray = data[i]
        rgba[i * 4 + 0] = gray
        rgba[i * 4 + 1] = gray
        rgba[i * 4 + 2] = gray
        rgba[i * 4 + 3] = 255
    
    return bytes(rgba)


def tex_format_to_dxt_flags(tex_format: TexFormat) -> Optional[DXTFlags]:
    """Convert TEX format to DXT flags"""
    if tex_format == TexFormat.DXT1:
        return DXTFlags.DXT1
    elif tex_format == TexFormat.DXT3:
        return DXTFlags.DXT3
    elif tex_format == TexFormat.DXT5:
        return DXTFlags.DXT5
    return None


def is_compressed_format(tex_format: TexFormat) -> bool:
    """Check if format is compressed"""
    return tex_format in (TexFormat.DXT1, TexFormat.DXT3, TexFormat.DXT5)


def is_raw_format(tex_format: TexFormat) -> bool:
    """Check if format is raw (uncompressed)"""
    return tex_format in (TexFormat.RGBA8888, TexFormat.RG88, TexFormat.R8)


def get_file_extension_from_mipmap_format(mipmap_format: MipmapFormat) -> str:
    """Get file extension based on mipmap format"""
    if mipmap_format == MipmapFormat.IMAGE_PNG:
        return '.png'
    elif mipmap_format == MipmapFormat.IMAGE_JPEG:
        return '.jpg'
    elif mipmap_format == MipmapFormat.IMAGE_BMP:
        return '.bmp'
    elif mipmap_format == MipmapFormat.IMAGE_GIF:
        return '.gif'
    elif mipmap_format == MipmapFormat.VIDEO_MP4:
        return '.mp4'
    else:
        # For compressed formats and raw formats, use PNG as output
        return '.png'


def get_mipmap_format(image_format: int, tex_format: TexFormat) -> MipmapFormat:
    """Determine mipmap format based on image_format and tex_format (mimics C# logic)"""
    # FIF_UNKNOWN is -1 in FreeImageFormat
    if image_format != -1:
        # For non-unknown image formats, map directly to MipmapFormat
        try:
            return MipmapFormat(image_format)
        except ValueError:
            logger.warning(f"Unknown image format: {image_format}, defaulting to PNG")
            return MipmapFormat.IMAGE_PNG
    
    # For FIF_UNKNOWN (-1), use tex_format
    if tex_format == TexFormat.RGBA8888:
        return MipmapFormat.RGBA8888
    elif tex_format == TexFormat.DXT5:
        return MipmapFormat.COMPRESSED_DXT5
    elif tex_format == TexFormat.DXT3:
        return MipmapFormat.COMPRESSED_DXT3
    elif tex_format == TexFormat.DXT1:
        return MipmapFormat.COMPRESSED_DXT1
    elif tex_format == TexFormat.R8:
        return MipmapFormat.R8
    elif tex_format == TexFormat.RG88:
        return MipmapFormat.RG88
    else:
        raise ValueError(f"Unsupported TEX format: {tex_format}")


def convert_tex_to_image(tex_data: bytes, mipmap_format: MipmapFormat, 
                        width: int, height: int) -> Tuple[bytes, MipmapFormat]:
    """Convert TEX data to standard image format"""
    # Handle non-unknown image formats (already in target format)
    if mipmap_format.value >= 0:
        # This is a FreeImageFormat value, data is already in the target format
        # No conversion needed, just return as-is
        return tex_data, mipmap_format
    
    # Handle compressed and raw formats (negative values)
    if mipmap_format == MipmapFormat.COMPRESSED_DXT5:
        rgba_data = decompress_dxt_image(width, height, tex_data, DXTFlags.DXT5)
        return rgba_data, MipmapFormat.RGBA8888
    elif mipmap_format == MipmapFormat.COMPRESSED_DXT3:
        rgba_data = decompress_dxt_image(width, height, tex_data, DXTFlags.DXT3)
        return rgba_data, MipmapFormat.RGBA8888
    elif mipmap_format == MipmapFormat.COMPRESSED_DXT1:
        rgba_data = decompress_dxt_image(width, height, tex_data, DXTFlags.DXT1)
        return rgba_data, MipmapFormat.RGBA8888
    elif mipmap_format == MipmapFormat.RGBA8888:
        # Already in RGBA8888 format
        expected_size = width * height * 4
        if len(tex_data) != expected_size:
            logger.warning(f"RGBA8888 data size mismatch: expected {expected_size}, got {len(tex_data)}")
            if len(tex_data) < expected_size:
                tex_data = tex_data + b'\x00' * (expected_size - len(tex_data))
            else:
                tex_data = tex_data[:expected_size]
        return tex_data, MipmapFormat.RGBA8888
    elif mipmap_format == MipmapFormat.RG88:
        rgba_data = convert_rg88_to_rgba8888(tex_data, width, height)
        return rgba_data, MipmapFormat.RGBA8888
    elif mipmap_format == MipmapFormat.R8:
        rgba_data = convert_r8_to_rgba8888(tex_data, width, height)
        return rgba_data, MipmapFormat.RGBA8888
    else:
        raise ValueError(f"Unsupported mipmap format for conversion: {mipmap_format}")


def save_as_png(rgba_data: bytes, width: int, height: int, output_path: str):
    """Save RGBA data as PNG"""
    # Create PIL Image from RGBA data
    img = Image.frombytes('RGBA', (width, height), rgba_data)
    img.save(output_path, 'PNG')


def read_tex_header(reader: BinaryIO) -> TexHeader:
    """Read TEX header"""
    header = TexHeader()
    header.format = TexFormat(struct.unpack('<i', reader.read(4))[0])
    header.flags = TexFlags(struct.unpack('<i', reader.read(4))[0])
    header.texture_width = struct.unpack('<i', reader.read(4))[0]
    header.texture_height = struct.unpack('<i', reader.read(4))[0]
    header.image_width = struct.unpack('<i', reader.read(4))[0]
    header.image_height = struct.unpack('<i', reader.read(4))[0]
    header.unk_int0 = struct.unpack('<I', reader.read(4))[0]  # unsigned int
    return header


def read_tex_mipmap(reader: BinaryIO, tex_format: TexFormat) -> TexMipmap:
    """Read TEX mipmap"""
    mipmap = TexMipmap()
    mipmap.width = struct.unpack('<i', reader.read(4))[0]
    mipmap.height = struct.unpack('<i', reader.read(4))[0]
    mipmap.format = tex_format
    
    # Read mipmap data
    data_size = struct.unpack('<i', reader.read(4))[0]
    mipmap.bytes = reader.read(data_size)
    
    return mipmap


@dataclass
class TexImageContainer:
    """Container for TEX images"""
    magic: str = ""
    image_count: int = 0
    image_format: int = 0
    image_container_version: int = 1


def read_tex_image_container(reader: BinaryIO) -> TexImageContainer:
    """Read TEX image container"""
    container = TexImageContainer()
    container.magic = read_n_string(reader, 16)
    
    logger.debug(f"TEXB magic: '{container.magic}'")
    
    container.image_count = struct.unpack('<i', reader.read(4))[0]
    
    # Handle different TEXB versions
    if container.magic == "TEXB0003":
        container.image_format = struct.unpack('<i', reader.read(4))[0]
        container.image_container_version = 3
    elif container.magic == "TEXB0004":
        container.image_format = struct.unpack('<i', reader.read(4))[0]
        is_video_mp4 = struct.unpack('<i', reader.read(4))[0] == 1
        if container.image_format == -1 and is_video_mp4:  # FIF_UNKNOWN with video
            container.image_format = 35  # FIF_MP4 equivalent
            container.image_container_version = 4
        else:
            # Non-MP4 TEXB0004 should be treated as Version 3, but keep the image_format
            container.image_container_version = 3
    elif container.magic in ["TEXB0001", "TEXB0002"]:
        container.image_format = -1  # FIF_UNKNOWN
        container.image_container_version = int(container.magic[4:])
    else:
        raise ValueError(f"Unknown TEXB magic: {container.magic}")
    
    return container


def read_tex_mipmap_v1(reader: BinaryIO) -> TexMipmap:
    """Read mipmap for TEXB0001"""
    mipmap = TexMipmap()
    mipmap.width = struct.unpack('<i', reader.read(4))[0]
    mipmap.height = struct.unpack('<i', reader.read(4))[0]
    
    data_size = struct.unpack('<i', reader.read(4))[0]
    if data_size < 0:
        raise ValueError(f"Invalid mipmap data size: {data_size}")
    mipmap.bytes = reader.read(data_size)
    
    return mipmap


def decompress_lz4(compressed_data: bytes, decompressed_size: int) -> bytes:
    """Decompress LZ4 compressed data"""
    try:
        logger.debug(f"LZ4 decompress: compressed_size={len(compressed_data)}, expected_decompressed_size={decompressed_size}")
        result = lz4.block.decompress(compressed_data, uncompressed_size=decompressed_size)
        logger.debug(f"LZ4 decompress success: actual_decompressed_size={len(result)}")
        return result
    except Exception as e:
        logger.error(f"LZ4 decompression failed: {e}")
        raise


def read_tex_mipmap_v2_v3(reader: BinaryIO) -> TexMipmap:
    """Read mipmap for TEXB0002/0003"""
    mipmap = TexMipmap()
    mipmap.width = struct.unpack('<i', reader.read(4))[0]
    mipmap.height = struct.unpack('<i', reader.read(4))[0]
    
    is_lz4_compressed = struct.unpack('<i', reader.read(4))[0] == 1
    decompressed_bytes_count = struct.unpack('<i', reader.read(4))[0]
    
    data_size = struct.unpack('<i', reader.read(4))[0]
    if data_size < 0:
        raise ValueError(f"Invalid mipmap data size: {data_size}")
    compressed_data = reader.read(data_size)
    
    logger.debug(f"Mipmap: width={mipmap.width}, height={mipmap.height}, is_lz4={is_lz4_compressed}, decompressed_size={decompressed_bytes_count}, compressed_size={data_size}")
    
    if is_lz4_compressed:
        mipmap.bytes = decompress_lz4(compressed_data, decompressed_bytes_count)
    else:
        mipmap.bytes = compressed_data
        logger.debug(f"Non-compressed mipmap data size: {len(mipmap.bytes)}")
    
    return mipmap


def read_tex_mipmap_v4(reader: BinaryIO) -> TexMipmap:
    """Read mipmap for TEXB0004"""
    # Read the additional parameters
    param1 = struct.unpack('<i', reader.read(4))[0]
    if param1 != 1:
        raise ValueError(f"Unexpected param1 value: {param1}")
    
    param2 = struct.unpack('<i', reader.read(4))[0]
    if param2 != 2:
        raise ValueError(f"Unexpected param2 value: {param2}")
    
    # Read condition JSON string
    condition_json = read_n_string(reader)
    
    param3 = struct.unpack('<i', reader.read(4))[0]
    if param3 != 1:
        raise ValueError(f"Unexpected param3 value: {param3}")
    
    # Now read the actual mipmap data
    mipmap = TexMipmap()
    mipmap.width = struct.unpack('<i', reader.read(4))[0]
    mipmap.height = struct.unpack('<i', reader.read(4))[0]
    
    is_lz4_compressed = struct.unpack('<i', reader.read(4))[0] == 1
    decompressed_bytes_count = struct.unpack('<i', reader.read(4))[0]
    
    data_size = struct.unpack('<i', reader.read(4))[0]
    if data_size < 0:
        raise ValueError(f"Invalid mipmap data size: {data_size}")
    compressed_data = reader.read(data_size)
    
    if is_lz4_compressed:
        mipmap.bytes = decompress_lz4(compressed_data, decompressed_bytes_count)
    else:
        mipmap.bytes = compressed_data
    
    return mipmap


def pick_mipmap_reader(container_version: int):
    """Pick appropriate mipmap reader based on container version"""
    if container_version == 1:
        return read_tex_mipmap_v1
    elif container_version in [2, 3]:
        return read_tex_mipmap_v2_v3
    elif container_version == 4:
        return read_tex_mipmap_v4
    else:
        raise ValueError(f"Unsupported TEX container version: {container_version}")


def read_tex_image(reader: BinaryIO, tex_format: TexFormat, container_version: int, image_format: int) -> TexImage:
    """Read TEX image"""
    image = TexImage()
    mipmap_count = struct.unpack('<i', reader.read(4))[0]
    
    read_mipmap_func = pick_mipmap_reader(container_version)
    
    # Determine the actual mipmap format
    actual_mipmap_format = get_mipmap_format(image_format, tex_format)
    logger.debug(f"Determined mipmap format: {actual_mipmap_format.name}")
    
    for _ in range(mipmap_count):
        mipmap = read_mipmap_func(reader)
        mipmap.format = actual_mipmap_format
        image.mipmaps.append(mipmap)
    
    return image


def read_tex_frame_info_container(reader: BinaryIO) -> TexFrameInfoContainer:
    """Read TEX frame info container for GIF animations"""
    container = TexFrameInfoContainer()
    container.magic = read_n_string(reader, 16)
    
    frame_count = struct.unpack('<i', reader.read(4))[0]
    
    # Read version-specific data
    if container.magic == "TEXS0003":
        container.gif_width = struct.unpack('<i', reader.read(4))[0]
        container.gif_height = struct.unpack('<i', reader.read(4))[0]
    
    # Read frames
    for _ in range(frame_count):
        frame = TexFrameInfo()
        frame.image_id = struct.unpack('<i', reader.read(4))[0]
        frame.frametime = struct.unpack('<f', reader.read(4))[0]
        
        if container.magic in ["TEXS0002", "TEXS0003"]:
            # Float coordinates for newer versions
            frame.x = struct.unpack('<f', reader.read(4))[0]
            frame.y = struct.unpack('<f', reader.read(4))[0]
            frame.width = struct.unpack('<f', reader.read(4))[0]
            frame.width_y = struct.unpack('<f', reader.read(4))[0]
            frame.height_x = struct.unpack('<f', reader.read(4))[0]
            frame.height = struct.unpack('<f', reader.read(4))[0]
        else:
            # Integer coordinates for older versions
            frame.x = float(struct.unpack('<i', reader.read(4))[0])
            frame.y = float(struct.unpack('<i', reader.read(4))[0])
            frame.width = float(struct.unpack('<i', reader.read(4))[0])
            frame.width_y = float(struct.unpack('<i', reader.read(4))[0])
            frame.height_x = float(struct.unpack('<i', reader.read(4))[0])
            frame.height = float(struct.unpack('<i', reader.read(4))[0])
        
        container.frames.append(frame)
    
    # Set default gif dimensions if not provided
    if container.gif_width == 0 or container.gif_height == 0 and container.frames:
        # Use first frame dimensions
        first_frame = container.frames[0]
        if first_frame.width != 0:
            container.gif_width = int(abs(first_frame.width))
            container.gif_height = int(abs(first_frame.height))
        else:
            container.gif_width = int(abs(first_frame.height_x))
            container.gif_height = int(abs(first_frame.width_y))
    
    return container


def read_n_string(reader: BinaryIO, max_length: int = -1) -> str:
    """Read null-terminated string"""
    builder = []
    while True:
        byte = reader.read(1)
        if not byte or byte == b'\x00':
            break
        if max_length > 0 and len(builder) >= max_length:
            break
        try:
            builder.append(byte.decode('utf-8'))
        except UnicodeDecodeError:
            # Skip invalid bytes
            continue
    
    return ''.join(builder)


def rotate_and_crop_frame(frame_img: Image.Image, frame_info: TexFrameInfo, 
                         target_width: int, target_height: int) -> Image.Image:
    """Rotate and crop frame according to frame info"""
    # Calculate bounding box coordinates
    x_coords = [
        frame_info.x,
        frame_info.x + frame_info.width,
        frame_info.x + frame_info.height_x,
        frame_info.x + frame_info.width + frame_info.height_x
    ]
    y_coords = [
        frame_info.y,
        frame_info.y + frame_info.height,
        frame_info.y + frame_info.width_y,
        frame_info.y + frame_info.height + frame_info.width_y
    ]
    
    left, right = min(x_coords), max(x_coords)
    top, bottom = min(y_coords), max(y_coords)
    
    # Ensure we have valid coordinates
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(frame_img.width, int(right)), min(frame_img.height, int(bottom))
    
    if left >= right or top >= bottom:
        # Create empty frame if coordinates are invalid
        return Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
    
    # Crop the frame
    crop_box = (left, top, right, bottom)
    cropped = frame_img.crop(crop_box)
    
    # Calculate rotation angle based on frame dimensions
    width_val = frame_info.width if frame_info.width != 0 else frame_info.height_x
    height_val = frame_info.height if frame_info.height != 0 else frame_info.width_y
    
    # Determine if rotation is needed
    if (frame_info.width == 0 or frame_info.height == 0) and (frame_info.height_x != 0 or frame_info.width_y != 0):
        # This indicates a rotated frame
        # Calculate rotation angle (simplified approach)
        if width_val > 0 and height_val < 0:
            # 90 degree clockwise rotation
            cropped = cropped.transpose(Image.ROTATE_270)
        elif width_val < 0 and height_val > 0:
            # 90 degree counter-clockwise rotation  
            cropped = cropped.transpose(Image.ROTATE_90)
        elif width_val < 0 and height_val < 0:
            # 180 degree rotation
            cropped = cropped.transpose(Image.ROTATE_180)
    
    # Resize to target dimensions if needed
    if cropped.size != (target_width, target_height):
        cropped = cropped.resize((target_width, target_height), Image.LANCZOS)
    
    return cropped


def create_gif_from_frames(frames: List[Image.Image], frame_times: List[float], output_path: str):
    """Create GIF from frames with specified frame times"""
    if not frames:
        return
    
    # Calculate frame durations in milliseconds (GIF uses 1/100th seconds)
    durations = []
    for time in frame_times:
        # Convert frametime to milliseconds, minimum 10ms
        duration = max(10, int(time * 1000))
        durations.append(duration)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,  # Infinite loop
        disposal=2  # Restore to background color
    )


def convert_tex_file(tex_path: str, overwrite: bool = False):
    """Convert a TEX file to PNG/GIF/MP4"""
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow is required for TEX conversion")
    
    try:
        with open(tex_path, 'rb') as f:
            # Read magic strings (null-terminated, not length-prefixed)
            magic1 = read_n_string(f, 16)
            if magic1 != "TEXV0005":
                raise ValueError(f"Invalid TEX magic1: {magic1}")
            
            magic2 = read_n_string(f, 16)
            if magic2 != "TEXI0001":
                raise ValueError(f"Invalid TEX magic2: {magic2}")
            
            # Read header
            header = read_tex_header(f)
            
            # Read image container
            image_container = read_tex_image_container(f)
            
            # Check if it's a video texture
            if (image_container.image_container_version == 4 and 
                image_container.image_format == 100):  # MP4 format
                # For video textures, the remaining data should be MP4
                mp4_data = f.read()
                if len(mp4_data) >= 12:
                    try:
                        mp4_magic = mp4_data[4:12].decode('ascii')
                        if mp4_magic in ['ftypisom', 'ftypmsnv', 'ftypmp42']:
                            mp4_path = os.path.splitext(tex_path)[0] + '.mp4'
                            if not overwrite and os.path.exists(mp4_path):
                                return
                            
                            with open(mp4_path, 'wb') as mp4_file:
                                mp4_file.write(mp4_data)
                            
                            json_info = {
                                'format': 'VIDEO_MP4',
                                'original_size': len(mp4_data),
                                'is_video_texture': True
                            }
                            json_path = os.path.splitext(tex_path)[0] + '.tex-json'
                            with open(json_path, 'w') as json_file:
                                json.dump(json_info, json_file, indent=2)
                            return
                    except (UnicodeDecodeError, IndexError):
                        pass
                
                # If not valid MP4, fall through to normal processing
            
            # Read images
            images = []
            for _ in range(image_container.image_count):
                image = read_tex_image(f, header.format, image_container.image_container_version, image_container.image_format)
                images.append(image)
            
            # Read frame info container if it's a GIF
            frame_info_container = None
            if header.is_gif:
                frame_info_container = read_tex_frame_info_container(f)
            
            if frame_info_container is not None:
                # Handle GIF animation
                gif_frames = []
                frame_times = []
                
                # Get base frame dimensions
                base_width = frame_info_container.gif_width
                base_height = frame_info_container.gif_height
                
                for frame_info in frame_info_container.frames:
                    if frame_info.image_id < len(images):
                        source_mipmap = images[frame_info.image_id].first_mipmap
                        if source_mipmap:
                            rgba_data, _ = convert_tex_to_image(
                                source_mipmap.bytes, 
                                source_mipmap.format,
                                source_mipmap.width,
                                source_mipmap.height
                            )
                            
                            # Create base image
                            base_img = Image.frombytes('RGBA', (source_mipmap.width, source_mipmap.height), rgba_data)
                            
                            # Rotate and crop according to frame info
                            processed_frame = rotate_and_crop_frame(base_img, frame_info, base_width, base_height)
                            gif_frames.append(processed_frame)
                            frame_times.append(frame_info.frametime)
                
                if gif_frames:
                    gif_path = os.path.splitext(tex_path)[0] + '.gif'
                    if not overwrite and os.path.exists(gif_path):
                        return
                    
                    create_gif_from_frames(gif_frames, frame_times, gif_path)
                    
                    # Create JSON info
                    json_info = {
                        'format': header.format.name,
                        'width': base_width,
                        'height': base_height,
                        'frame_count': len(gif_frames),
                        'is_gif': True,
                        'original_size': sum(len(img.first_mipmap.bytes) for img in images if img.first_mipmap)
                    }
                    json_path = os.path.splitext(tex_path)[0] + '.tex-json'
                    with open(json_path, 'w') as json_file:
                        json.dump(json_info, json_file, indent=2)
            
            else:
                # Handle single image
                if images and images[0].first_mipmap:
                    source_mipmap = images[0].first_mipmap
                    rgba_data, output_format = convert_tex_to_image(
                        source_mipmap.bytes,
                        source_mipmap.format,  # Use the determined mipmap format
                        source_mipmap.width,
                        source_mipmap.height
                    )
                    
                    # Determine output file extension
                    output_ext = get_file_extension_from_mipmap_format(output_format)
                    output_path = os.path.splitext(tex_path)[0] + output_ext
                    
                    if not overwrite and os.path.exists(output_path):
                        return
                    
                    # Save the data
                    if output_format.value < 0:  # Raw and compressed formats (negative values)
                        # These need to be saved as PNG
                        save_as_png(rgba_data, source_mipmap.width, source_mipmap.height, output_path)
                    else:
                        # Directly save the data (already in target format)
                        with open(output_path, 'wb') as out_file:
                            out_file.write(rgba_data)
                    
                    # Create JSON info
                    json_info = {
                        'format': header.format.name,
                        'output_format': output_format.name,
                        'width': header.image_width,
                        'height': header.image_height,
                        'texture_width': header.texture_width,
                        'texture_height': header.texture_height,
                        'flags': header.flags,
                        'original_size': len(source_mipmap.bytes),
                        'is_direct_save': output_format in [MipmapFormat.IMAGE_PNG, MipmapFormat.IMAGE_JPEG, 
                                                         MipmapFormat.IMAGE_BMP, MipmapFormat.IMAGE_GIF, 
                                                         MipmapFormat.VIDEO_MP4]
                    }
                    json_path = os.path.splitext(tex_path)[0] + '.tex-json'
                    with open(json_path, 'w') as json_file:
                        json.dump(json_info, json_file, indent=2)
    
    except Exception as e:
        logger.error(f"Failed to process TEX file {tex_path}: {e}")
        raise


def extract_pkg_file(pkg_path: str, output_dir: str, options: argparse.Namespace, stats: ExtractionStats = None):
    """Extract a single PKG file"""
    logger.info(f"\n### Extracting package: {pkg_path}")
    
    # Read PKG file
    with open(pkg_path, 'rb') as f:
        reader = PackageReader(read_entry_bytes=True)
        package = reader.read_from(f)
    
    # Get output directory
    output_directory = output_dir
    
    # Filter entries based on options
    entries = package.entries
    if hasattr(options, 'ignoreexts') and options.ignoreexts:
        skip_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in options.ignoreexts.split(',')]
        entries = [e for e in entries if not any(e.full_path.lower().endswith(ext.lower()) for ext in skip_exts)]
    
    if hasattr(options, 'onlyexts') and options.onlyexts:
        only_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in options.onlyexts.split(',')]
        entries = [e for e in entries if any(e.full_path.lower().endswith(ext.lower()) for ext in only_exts)]
    
    # Extract entries
    for entry in entries:
        extract_entry(entry, output_directory, options, stats)


def extract_entry(entry: PackageEntry, output_dir: str, options: argparse.Namespace, stats: ExtractionStats = None):
    """Extract a single package entry"""
    if stats:
        stats.add_total()
    
    # Determine output path
    if hasattr(options, 'singledir') and options.singledir:
        output_path = os.path.join(output_dir, entry.name + entry.extension)
    else:
        output_path = os.path.join(output_dir, entry.directory_path, entry.name + entry.extension)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Skip if file exists and overwrite is not enabled
    if not (hasattr(options, 'overwrite') and options.overwrite) and os.path.exists(output_path):
        logger.info(f"* Skipping, already exists: {output_path}")
        if stats:
            stats.add_success()  # Count skipped files as success
        return
    
    logger.info(f"* Extracting: {entry.full_path}")
    
    try:
        # Write raw file
        with open(output_path, 'wb') as f:
            f.write(entry.bytes)
        
        # Convert TEX files if requested
        if not (hasattr(options, 'no_tex_convert') and options.no_tex_convert) and entry.type == EntryType.TEX:
            convert_tex_file(output_path, hasattr(options, 'overwrite') and options.overwrite)
        
        if stats:
            stats.add_success()
    except Exception as e:
        logger.error(f"Failed to process {entry.full_path}: {e}")
        if stats:
            stats.add_failed()


def extract_directory(input_dir: str, output_dir: str, options: argparse.Namespace, stats: ExtractionStats = None):
    """Extract all PKG/TEX files from a directory"""
    input_path = Path(input_dir)
    
    if hasattr(options, 'tex') and options.tex:
        # Convert all TEX files in directory
        pattern = "**/*.tex" if (hasattr(options, 'recursive') and options.recursive) else "*.tex"
        for tex_file in input_path.glob(pattern):
            if tex_file.is_file():
                if stats:
                    stats.add_total()
                try:
                    convert_tex_file(str(tex_file), hasattr(options, 'overwrite') and options.overwrite)
                    if stats:
                        stats.add_success()
                except Exception as e:
                    logger.error(f"Failed to process {tex_file}: {e}")
                    if stats:
                        stats.add_failed()
    else:
        # Extract all PKG files
        pattern = "**/*.pkg" if (hasattr(options, 'recursive') and options.recursive) else "*.pkg"
        for pkg_file in input_path.glob(pattern):
            if pkg_file.is_file():
                try:
                    extract_pkg_file(str(pkg_file), output_dir, options, stats)
                except Exception as e:
                    logger.error(f"Failed to extract {pkg_file}: {e}")
                    if stats:
                        stats.add_failed()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Wallpaper Engine PKG Unpacker - Extract and convert Wallpaper Engine files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported operations:
  extract  - Extract PKG files and convert TEX to images
  info     - Display file information

Examples:
  %(prog)s extract sample.pkg -o ./output
  %(prog)s extract sample.pkg --no-tex-convert -o ./output
  %(prog)s extract ./textures -t -r --overwrite
  %(prog)s info sample.pkg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract PKG files and convert TEX to images')
    extract_parser.add_argument('input', help='Path to PKG file or directory')
    extract_parser.add_argument('-o', '--output', default='./output', help='Output directory (default: ./output)')
    extract_parser.add_argument('-i', '--ignoreexts', help='Ignore files with specified extensions (comma separated)')
    extract_parser.add_argument('-e', '--onlyexts', help='Only extract files with specified extensions (comma separated)')
    extract_parser.add_argument('-t', '--tex', action='store_true', help='Convert all TEX files in directory to images')
    extract_parser.add_argument('-s', '--singledir', action='store_true', help='Put all extracted files in one directory')
    extract_parser.add_argument('-r', '--recursive', action='store_true', help='Recursive search in subdirectories')
    extract_parser.add_argument('--no-tex-convert', action='store_true', help='Do not convert TEX files to images')
    extract_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display file information')
    info_parser.add_argument('input', help='Path to PKG or TEX file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate PIL availability for image operations
    if args.command == 'extract' and not PIL_AVAILABLE:
        logger.error("Error: PIL/Pillow is required for image operations")
        logger.error("Install with: pip install Pillow")
        return
    
    # Validate LZ4 availability for TEX operations
    if args.command == 'extract' and not LZ4_AVAILABLE:
        logger.warning("Warning: lz4 library not available. Some TEX compression features may not work")
        logger.warning("Install with: pip install lz4")
    
    try:
        if args.command == 'extract':
            input_path = Path(args.input)
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize statistics
            stats = ExtractionStats()
            
            if input_path.is_file():
                if input_path.suffix.lower() == '.pkg':
                    extract_pkg_file(str(input_path), str(output_path), args, stats)
                elif input_path.suffix.lower() == '.tex':
                    stats.add_total()
                    try:
                        convert_tex_file(str(input_path), args.overwrite)
                        stats.add_success()
                    except Exception as e:
                        logger.error(f"Failed to process {input_path}: {e}")
                        stats.add_failed()
                else:
                    logger.error(f"Unsupported file type: {input_path}")
                    return
            elif input_path.is_dir():
                extract_directory(str(input_path), str(output_path), args, stats)
            else:
                logger.error(f"Input not found: {args.input}")
                return
            
            # Print summary
            stats.print_summary()
        
        elif args.command == 'info':
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"File not found: {input_path}")
                return
            
            if input_path.suffix.lower() == '.pkg':
                with open(input_path, 'rb') as f:
                    reader = PackageReader(read_entry_bytes=False)
                    package = reader.read_from(f)
                    logger.info(f"PKG File: {input_path}")
                    logger.info(f"Magic: {package.magic}")
                    logger.info(f"Header Size: {package.header_size}")
                    logger.info(f"Entries: {len(package.entries)}")
                    for entry in package.entries:
                        logger.info(f"  {entry.full_path} ({entry.length} bytes)")
            
            elif input_path.suffix.lower() == '.tex':
                with open(input_path, 'rb') as f:
                    magic1 = read_n_string(f, 16)
                    magic2 = read_n_string(f, 16)
                    header = read_tex_header(f)
                    logger.info(f"TEX File: {input_path}")
                    logger.info(f"Magic1: {magic1}")
                    logger.info(f"Magic2: {magic2}")
                    logger.info(f"Format: {header.format.name}")
                    logger.info(f"Flags: {header.flags}")
                    logger.info(f"Dimensions: {header.image_width}x{header.image_height}")
                    logger.info(f"Texture Dimensions: {header.texture_width}x{header.texture_height}")
                    logger.info(f"Is GIF: {header.is_gif}")
                    logger.info(f"Is Video Texture: {header.is_video_texture}")
            
            else:
                logger.error(f"Unsupported file type for info: {input_path.suffix}")
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()