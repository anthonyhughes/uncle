#!/usr/bin/env python3
"""Read-only checks against original figure PDFs and the submitted PDF.

The submitted document embeds figures as Form XObjects. Their content streams
and complete rendering-resource graphs are compared with the standalone
reference figures after decoding transport compression and resolving indirect
object IDs. Fonts, glyph programs, marker paths, images, and graphics states
remain part of the comparison; equality of drawing commands alone is not enough.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from PyPDF2 import PdfReader
from PyPDF2.generic import (
    ArrayObject, BooleanObject, ByteStringObject, DictionaryObject,
    FloatObject, IndirectObject, NameObject, NullObject, NumberObject,
    StreamObject, TextStringObject,
)


def digest(path: Path) -> str:
    """SHA-256 of the complete file, without loading large files all at once."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _single_page(path: Path):
    reader = PdfReader(str(path))
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a single-page standalone figure: {path}")
    return reader.pages[0]


def pdf_content_sha256(path: Path) -> str:
    """SHA-256 of one standalone figure's decoded page drawing commands."""
    contents = _single_page(Path(path)).get_contents()
    if contents is None:
        raise ValueError(f"PDF has no page drawing commands: {path}")
    return hashlib.sha256(contents.get_data()).hexdigest()


def _canonical(value: Any, active: frozenset[int] = frozenset(), *,
               round_resource_bboxes: bool = False) -> Any:
    """Resolve the semantic resource graph, retaining all rendering payloads.

    PDF object numbering, dictionary ordering, stream length and compression
    representation are transport details. Everything else is retained.
    Cyclic graphs fail explicitly rather than being silently declared equal.
    """
    if isinstance(value, IndirectObject):
        value = value.get_object()
    if isinstance(value, (StreamObject, DictionaryObject, ArrayObject)):
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic PDF resource graph cannot be canonicalized")
        nested = active | {identity}
        if isinstance(value, ArrayObject):
            return ["array", [_canonical(item, nested, round_resource_bboxes=round_resource_bboxes)
                              for item in value]]
        ignored = {"/Length", "/Filter", "/DecodeParms"} if isinstance(value, StreamObject) else set()
        entries = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if str(key) in ignored:
                continue
            if round_resource_bboxes and str(key) == "/BBox":
                # pdfTeX's inclusion rewrites these rectangle coordinates to
                # six decimal places. Do not round drawing commands, fonts,
                # glyph programs, matrices, or any other numerical resources.
                coordinates = item.get_object() if isinstance(item, IndirectObject) else item
                encoded = ["array", [
                    _canonical(Decimal(str(coordinate)).quantize(Decimal("0.000001")))
                    for coordinate in coordinates
                ]]
            else:
                encoded = _canonical(item, nested, round_resource_bboxes=round_resource_bboxes)
            entries.append([str(key), encoded])
        if isinstance(value, StreamObject):
            decoded = value.get_data()
            return ["stream", entries, len(decoded), hashlib.sha256(decoded).hexdigest()]
        return ["dict", entries]
    if isinstance(value, NameObject):
        return ["name", str(value)]
    if isinstance(value, TextStringObject):
        return ["text", str(value)]
    if isinstance(value, (ByteStringObject, bytes)):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, BooleanObject):
        return ["bool", bool(value.value)]
    if isinstance(value, NullObject) or value is None:
        return ["null"]
    if isinstance(value, (NumberObject, FloatObject, int, float, Decimal)):
        number = Decimal(str(value))
        return ["number", "0" if number == 0 else str(number.normalize())]
    raise TypeError(f"Unsupported PDF resource value: {type(value).__name__}")


def _fingerprint(value: Any, *, round_resource_bboxes: bool = False) -> str:
    encoded = json.dumps(_canonical(value, round_resource_bboxes=round_resource_bboxes),
                         ensure_ascii=True,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def compare_embedded(
    submission_pdf: Path,
    page_index: int,
    xobject_name: str,
    reference_pdf: Path,
) -> dict[str, Any]:
    """Compare a submitted Form XObject with its single-page reference figure.

    ``page_index`` is zero-based. Bounding boxes are reported both exactly and
    within 1e-4 point because pdfTeX rounds embedded decimal dimensions. This
    tolerance is never applied to content streams. Resource fingerprints are
    reported both exactly and after *only* rounding /BBox coordinates to six
    decimals, matching pdfTeX's observed import rewrite of marker rectangles.
    """
    submission = PdfReader(str(submission_pdf))
    page = submission.pages[page_index]
    resources = page["/Resources"].get_object()
    xobjects = resources["/XObject"].get_object()
    name = xobject_name if xobject_name.startswith("/") else "/" + xobject_name
    embedded = xobjects[name].get_object()
    if embedded.get("/Subtype") != "/Form":
        raise ValueError(f"{name} on page {page_index} is not a Form XObject")
    reference = _single_page(Path(reference_pdf))
    embedded_content = embedded.get_data()
    reference_content = reference.get_contents().get_data()
    embedded_resources = embedded.get("/Resources", DictionaryObject())
    reference_resources = reference.get("/Resources", DictionaryObject())
    embedded_resource_hash = _fingerprint(embedded_resources)
    reference_resource_hash = _fingerprint(reference_resources)
    embedded_import_hash = _fingerprint(embedded_resources, round_resource_bboxes=True)
    reference_import_hash = _fingerprint(reference_resources, round_resource_bboxes=True)
    bbox = [float(value) for value in embedded["/BBox"]]
    media_box = [float(value) for value in reference.mediabox]
    bbox_error = max(abs(left - right) for left, right in zip(bbox, media_box))
    matrix = [float(value) for value in embedded.get("/Matrix", [1, 0, 0, 1, 0, 0])]
    result = {
        "submission_pdf": str(submission_pdf),
        "reference_pdf": str(reference_pdf),
        "page_index": int(page_index),
        "xobject_name": name,
        "drawing_commands_match": embedded_content == reference_content,
        "embedded_drawing_sha256": hashlib.sha256(embedded_content).hexdigest(),
        "reference_drawing_sha256": hashlib.sha256(reference_content).hexdigest(),
        "resources_match": embedded_resource_hash == reference_resource_hash,
        "embedded_resources_sha256": embedded_resource_hash,
        "reference_resources_sha256": reference_resource_hash,
        "resources_match_after_bbox_import_rounding": embedded_import_hash == reference_import_hash,
        "embedded_resources_bbox6_sha256": embedded_import_hash,
        "reference_resources_bbox6_sha256": reference_import_hash,
        "resource_normalization": (
            "Resolve indirect IDs and decode compression; separately compare with "
            "only /BBox coordinates rounded to six decimal places, as pdfTeX imports them. "
            "No font, glyph, drawing-stream, or other numeric tolerance."
        ),
        "embedded_bbox_points": bbox,
        "reference_media_box_points": media_box,
        "bbox_exact_match": bbox == media_box,
        "bbox_max_absolute_error_points": bbox_error,
        "bbox_matches_within_0.0001_point": bbox_error <= .0001,
        "embedded_matrix": matrix,
        "embedded_matrix_is_identity": matrix == [1., 0., 0., 1., 0., 0.],
    }
    result["verified_match"] = bool(
        result["drawing_commands_match"]
        and result["resources_match_after_bbox_import_rounding"]
        and result["bbox_matches_within_0.0001_point"]
        and result["embedded_matrix_is_identity"]
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-pdf", type=Path, required=True)
    parser.add_argument("--page-index", type=int, required=True,
                        help="Zero-based submitted PDF page index")
    parser.add_argument("--xobject-name", required=True)
    parser.add_argument("--reference-pdf", type=Path, required=True)
    args = parser.parse_args()
    record = compare_embedded(args.submission_pdf, args.page_index,
                              args.xobject_name, args.reference_pdf)
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["verified_match"] else 1)
