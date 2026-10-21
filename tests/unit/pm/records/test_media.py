"""Record images: byte checks, header sizes, tags, and the typed metadata read."""
from __future__ import annotations

import base64
import struct
import unittest
import zlib

from pm.constants import MEDIA_MAX_IMAGE_BYTES
from pm.records.enums import FeatureType
from pm.records.media import (MEDIA_KEY, MediaProvenance, RecordMedia, alt_text_view, attach_images, image_from_base64,
                              image_from_bytes, image_ids, image_tag, merge_alt_text, metadata_media, record_media,
                              with_tags)
from pm.records.memory_records import Feature


def png(width: int, height: int) -> bytes:
    """Return a real RGB PNG of the given size, filled with grey."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + b"\x80\x80\x80" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


#: VP8X header of a 300x200 WebP (the canvas size fields are size minus one).
WEBP = b"RIFF" + struct.pack("<I", 30) + b"WEBPVP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" \
    + (299).to_bytes(3, "little") + (199).to_bytes(3, "little")
#: SOI, APP0 (JFIF), SOF0 of a 640x480 JPEG.
JPEG = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, 480, 640, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01")
GIF = b"GIF89a" + struct.pack("<HH", 32, 16) + b"\x00\x00\x00;"


class ImageFromBytesTests(unittest.TestCase):
    def test_magic_bytes_set_the_type_and_the_header_sets_the_size(self):
        cases = [(png(3, 2), "image/png", (3, 2)), (WEBP, "image/webp", (300, 200)),
                 (JPEG, "image/jpeg", (640, 480)), (GIF, "image/gif", (32, 16))]
        for data, media_type, size in cases:
            with self.subTest(media_type=media_type):
                image = image_from_bytes(data)
                self.assertEqual((image.media_type, (image.width, image.height)), (media_type, size))
                self.assertEqual(image.size_bytes, len(data))
                self.assertEqual(image.data(), data)
                self.assertTrue(image.id.startswith("img-") and image.id[4:] == image.sha256[:16])
                self.assertEqual(image.data_url(), f"data:{media_type};base64," + base64.b64encode(data).decode())

    def test_junk_a_wrong_type_and_oversized_data_are_rejected(self):
        for data in (b"", b"hello world, not an image", b"RIFF\x00\x00\x00\x00WAVEfmt "):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    image_from_bytes(data)
        with self.assertRaises(ValueError):
            image_from_bytes(png(1, 1), "image/jpeg")
        with self.assertRaises(ValueError):
            image_from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * MEDIA_MAX_IMAGE_BYTES)
        with self.assertRaises(ValueError):
            image_from_base64("not base64 !!!")

    def test_same_bytes_get_the_same_id(self):
        self.assertEqual(image_from_bytes(png(2, 2)).id, image_from_bytes(png(2, 2)).id)
        self.assertNotEqual(image_from_bytes(png(2, 2)).id, image_from_bytes(png(2, 3)).id)

    def test_a_supplied_alt_text_counts_as_user_text(self):
        image = image_from_bytes(png(1, 1), alt_text=" a grey square ")
        self.assertEqual((image.alt_text, image.alt_text_source), ("a grey square", "user"))


class AttachAndReadTests(unittest.TestCase):
    def test_attach_returns_tags_and_the_typed_read_returns_the_images(self):
        first, second = image_from_bytes(png(1, 1)), image_from_bytes(GIF, provenance=MediaProvenance(source="tool"))
        metadata: dict = {"session_id": "s1"}
        tags = attach_images(metadata, [first, second])
        self.assertEqual(tags, [first.tag, second.tag])
        tags_again = attach_images(metadata, [first])
        self.assertEqual(tags_again, [first.tag])
        feature = Feature(content=with_tags("look", tags), feature_type=FeatureType.Dialogue, metadata=metadata)
        media = record_media(feature)
        self.assertEqual(media.schema_id, "pm.record-media.v1")
        self.assertEqual([image.id for image in media.images], [first.id, second.id])
        self.assertEqual(media.images[1].provenance.source, "tool")
        self.assertEqual(feature.content, f"look\n{first.tag}\n{second.tag}")
        self.assertEqual(record_media(Feature(content="x", feature_type=FeatureType.Dialogue)), RecordMedia())
        self.assertEqual(with_tags("", tags), "\n".join(tags))

    def test_tags_parse_in_order_and_alt_view_replaces_them(self):
        image = image_from_bytes(png(1, 1), alt_text="a grey square")
        text = f"before {image.tag} middle {image_tag('img-missing')} {image.tag}"
        self.assertEqual(image_ids(text), [image.id, "img-missing", image.id])
        lookup = {image.id: image}.get
        self.assertEqual(alt_text_view(text, lookup),
                         "before [image: a grey square] middle [image: unavailable] [image: a grey square]")
        bare = image_from_bytes(png(1, 1))
        self.assertEqual(alt_text_view(bare.tag, {bare.id: bare}.get), "[image: no description yet]")

    def test_merge_keeps_a_newer_alt_text(self):
        image = image_from_bytes(png(1, 1))
        stale: dict = {}
        attach_images(stale, [image])
        current: dict = {}
        attach_images(current, [image.model_copy(update={"alt_text": "grey", "alt_text_source": "model"})])
        merge_alt_text(stale, current)
        self.assertEqual(metadata_media(stale).images[0].alt_text, "grey")
        self.assertIn(MEDIA_KEY, stale)


if __name__ == "__main__":
    unittest.main()
