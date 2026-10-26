#!/usr/bin/env python3
"""Regenerate and verify the submission figures selected by manifest.json.

Rendering uses explicitly pinned interpreters. All scientific inputs and
saved references are hash-checked before any renderer runs; output must be a
new or empty directory outside the authored workspace and frozen inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

# Also prevent cache writes if the caller forgot Python's -B flag.
sys.dont_write_bytecode = True

from common import prepare_output
from verify_submission import compare_embedded, digest, pdf_content_sha256


PACKAGE = Path(__file__).resolve().parent
EXPECTED_PYTHON = {"legacy": "3.10.12", "modern": "3.13.7", "figure2": "3.13.5"}
EXPECTED_MATPLOTLIB = {"legacy": "3.7.3", "modern": "3.11.1", "figure2": "3.10.8"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _requirements(profile: str) -> dict[str, str]:
    path = PACKAGE / f"requirements-{profile}.txt"
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"Expected exact dependency pins in {path}: {line}")
        name, version = line.split("==")
        versions[name.strip()] = version.strip()
    if versions.get("matplotlib") != EXPECTED_MATPLOTLIB[profile]:
        raise ValueError(f"Unexpected {profile} Matplotlib pin")
    return versions


def _environment(interpreter: str, profile: str, env: dict[str, str]) -> dict:
    if not interpreter:
        raise ValueError(f"The requested figure needs --{profile}-python")
    required = _requirements(profile)
    probe = (
        "import importlib.metadata as m,json,platform,sys; "
        "import matplotlib,matplotlib.ft2font; "
        "names=json.loads(sys.argv[1]); "
        "print(json.dumps(dict(python=platform.python_version(),"
        "implementation=platform.python_implementation(),executable=sys.executable,"
        "platform=platform.platform(),matplotlib=matplotlib.__version__,"
        "freetype=matplotlib.ft2font.__freetype_version__,"
        "dependencies={name:m.version(name) for name in names})))"
    )
    command = [interpreter, "-B", "-c", probe, json.dumps(list(required))]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    record = {
        "profile": profile, "interpreter_argument": interpreter,
        "required_python": EXPECTED_PYTHON[profile], "required_dependencies": required,
        "returncode": completed.returncode, "stdout": completed.stdout,
        "stderr": completed.stderr, "verified": False,
    }
    if completed.returncode:
        return record
    actual = json.loads(completed.stdout)
    record["actual"] = actual
    record["mismatches"] = {
        name: {"expected": version, "actual": actual["dependencies"].get(name)}
        for name, version in required.items()
        if actual["dependencies"].get(name) != version
    }
    if actual["python"] != EXPECTED_PYTHON[profile]:
        record["mismatches"]["python"] = {
            "expected": EXPECTED_PYTHON[profile], "actual": actual["python"]}
    if actual["implementation"] != "CPython":
        record["mismatches"]["implementation"] = {
            "expected": "CPython", "actual": actual["implementation"]}
    record["verified"] = not record["mismatches"]
    return record


def _check_file(path: Path, expected: str, role: str) -> dict:
    record = {"path": str(path), "role": role, "expected_sha256": expected,
              "actual_sha256": None, "verified": False}
    try:
        record["actual_sha256"] = digest(path)
        record["verified"] = record["actual_sha256"] == expected
    except (OSError, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _profile(entry: dict) -> str:
    matches = [name for name, version in EXPECTED_MATPLOTLIB.items()
               if entry["matplotlib"] == version]
    if len(matches) != 1:
        raise ValueError(f"No frozen environment for Figure {entry['number']}")
    return matches[0]


def _outputs(entry: dict) -> list[dict]:
    """Allow a composite figure to declare multiple independently verified PDFs.

    Optional ``outputs`` records use the same reference/hash/page/XObject fields
    as single-output figure records, plus an ``output_stem`` (e.g. figure_2a).
    ``reference_pdf`` may replace ``reference_stem`` when filenames are irregular.
    Omit ``png_sha256`` when no original PNG exists; its newly generated hash is
    still recorded, but is explicitly not called a byte-identical reproduction.
    """
    outputs = entry.get("outputs")
    if outputs is None:
        return [dict(entry, output_stem=f"figure_{entry['number']}")]
    if not outputs:
        raise ValueError(f"Figure {entry['number']} declares no outputs")
    stems = [item["output_stem"] for item in outputs]
    if len(set(stems)) != len(stems):
        raise ValueError("Composite figure output names must be distinct")
    for stem in stems:
        if Path(stem).name != stem or stem in {"", ".", ".."}:
            raise ValueError("Output stems must be plain filenames")
    return outputs


def _reference_path(root: Path, output: dict, extension: str) -> Path:
    relative = output.get(f"reference_{extension}")
    if relative is None:
        relative = output["reference_stem"] + "." + extension
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Reference escapes the external artifact root: {path}")
    return path


def _output_check(root: Path, out: Path, output: dict, extension: str, number: int) -> dict:
    path = out / f"{output['output_stem']}.{extension}"
    role = f"regenerated Figure {number} {extension}"
    expected = output.get(f"{extension}_sha256")
    if expected is not None:
        record = _check_file(path, expected, role)
        record["bytes_match"] = record["verified"]
        record["verification_status"] = (
            "original_byte_match" if record["verified"] else "original_byte_mismatch")
        if extension == "png":
            comparison = output.get("png_comparison", "bytes")
            if comparison not in {"bytes", "pixels"}:
                raise ValueError(f"Unsupported PNG comparison policy: {comparison}")
            record["comparison"] = comparison
            if comparison == "pixels":
                from PIL import Image

                original_path = _reference_path(root, output, "png")
                with Image.open(original_path) as original, Image.open(path) as generated:
                    original.load()
                    generated.load()
                    original_rgba = original.convert("RGBA").tobytes()
                    generated_rgba = generated.convert("RGBA").tobytes()
                    pixels_match = bool(
                        original.mode == generated.mode and original.size == generated.size
                        and original_rgba == generated_rgba)
                    record["pixel_details"] = {
                        "original_path": str(original_path),
                        "original_mode": original.mode, "generated_mode": generated.mode,
                        "original_size": list(original.size), "generated_size": list(generated.size),
                        "original_rgba_sha256": hashlib.sha256(original_rgba).hexdigest(),
                        "generated_rgba_sha256": hashlib.sha256(generated_rgba).hexdigest(),
                        "decoder_pillow": importlib.metadata.version("Pillow"),
                    }
                record["pixels_match"] = pixels_match
                record["verified"] = pixels_match
                record["verification_status"] = (
                    "original_pixel_match" if pixels_match else "original_pixel_mismatch")
                record["note"] = (
                    "Manifest requests exact decoded mode, size, and RGBA pixel equality. "
                    "The bytes_match field separately records whether PNG encoding also matches.")
        return record
    if extension != "png":
        raise ValueError("Every submitted panel must have an expected original PDF hash")
    if output.get("png_comparison", "bytes") == "pixels":
        raise ValueError("Pixel comparison requires an original PNG hash and path")
    return {
        "path": str(path), "role": role, "expected_sha256": None,
        "actual_sha256": digest(path), "verified": None,
        "verification_status": "no_original",
        "note": "Generated companion PNG exists; no original PNG was supplied for byte comparison.",
    }


def run(args: argparse.Namespace) -> int:
    manifest_path = PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["number"]: entry for entry in manifest["figures"]}
    if len(entries) != len(manifest["figures"]):
        raise ValueError("Manifest contains duplicate figure numbers")
    supported = tuple(sorted(entries))
    requested = supported if args.figures is None else tuple(args.figures)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(supported):
        raise ValueError(f"Choose distinct supported figures from {supported}")
    output_stems = [output["output_stem"] for number in requested for output in _outputs(entries[number])]
    if len({stem.casefold() for stem in output_stems}) != len(output_stems):
        raise ValueError("Requested figures have colliding output filenames")
    artifact_root = args.artifact_root.resolve()
    out = args.out_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be new or empty; previous results are never overwritten")
    out = prepare_output(artifact_root, out)
    receipt_path = out / "verification.json"
    receipt = {
        "schema_version": 1, "status": "in_progress", "started_utc": _utc(),
        "artifact_root": str(artifact_root), "output_directory": str(out),
        "submission_pdf": str(args.submission_pdf.resolve()),
        "requested_figures": list(requested), "supported_figures": list(supported),
        "requested_all_supported_figures": set(requested) == set(supported),
        "skipped_figures": manifest.get("skipped_figures", {}),
        "requested_coverage_complete": False, "all_supported_figures_verified": False,
        "preflight": [], "environments": {}, "figures": [],
        "runner_python": sys.version,
        "verifier_pypdf2": importlib.metadata.version("PyPDF2"),
        "source_sha256": {
            path.name: digest(path)
            for path in sorted(PACKAGE.iterdir())
            if path.is_file() and (path.suffix in {".py", ".json"} or path.name.startswith("requirements-"))
        },
        "timing_note": "Historical PDF creation metadata is preserved; started_utc/completed_utc identify this reproduction.",
    }
    try:
        if receipt["verifier_pypdf2"] != "3.0.1":
            raise ValueError("The verification interpreter must use PyPDF2 3.0.1")
        receipt["preflight"].append(_check_file(
            args.submission_pdf.resolve(), manifest["submission_sha256"], "submitted paper"))
        files = {}

        def add_preflight_file(path: Path, expected: str, role: str) -> None:
            if path in files and files[path][0] != expected:
                raise ValueError(f"Conflicting manifest hashes for {path}")
            files[path] = (expected, role)

        for number in requested:
            entry = entries[number]
            for relative, expected in entry["inputs"].items():
                path = (artifact_root / relative).resolve()
                if not path.is_relative_to(artifact_root):
                    raise ValueError(f"Input escapes the external artifact root: {relative}")
                add_preflight_file(path, expected, "frozen plotting input")
            for output in _outputs(entry):
                for extension in ("pdf", "png"):
                    if extension == "png" and output.get("png_sha256") is None:
                        continue
                    path = _reference_path(artifact_root, output, extension)
                    add_preflight_file(path, output[f"{extension}_sha256"],
                                       f"Figure {number} original {extension}")
        receipt["preflight"].extend(
            _check_file(path, expected, role)
            for path, (expected, role) in sorted(files.items(), key=lambda item: str(item[0]))
        )
        if not all(item["verified"] for item in receipt["preflight"]):
            raise ValueError("Input, original-figure, or submission hash mismatch; rendering was not started")

        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                   MPLCONFIGDIR=str(out / "_mplconfig"))
        interpreters = {"legacy": args.legacy_python, "modern": args.modern_python,
                        "figure2": args.figure2_python}
        profiles = {_profile(entries[number]) for number in requested}
        for profile in sorted(profiles):
            receipt["environments"][profile] = _environment(interpreters[profile], profile, env)
        if not all(item["verified"] for item in receipt["environments"].values()):
            raise ValueError("Pinned plotting environment mismatch; rendering was not started")

        for number in requested:
            entry = entries[number]
            profile = _profile(entry)
            command = [interpreters[profile], "-B", str(PACKAGE / f"figure_{number}.py"),
                       "--artifact-root", str(artifact_root), "--out-dir", str(out)]
            record = {"number": number, "profile": profile, "command": command,
                      "started_utc": _utc(), "verified": False}
            receipt["figures"].append(record)
            try:
                print(f"Rendering Figure {number} with {profile} environment...", flush=True)
                completed = subprocess.run(command, env=env, text=True,
                                           capture_output=True, check=False)
                record.update(returncode=completed.returncode, stdout=completed.stdout,
                              stderr=completed.stderr)
                if completed.returncode:
                    raise RuntimeError(f"Figure {number} renderer exited {completed.returncode}")
                record["outputs"] = []
                for output in _outputs(entry):
                    stem = output["output_stem"]
                    checks = [
                        _output_check(artifact_root, out, output, extension, number)
                        for extension in ("pdf", "png")
                    ]
                    pdf = out / f"{stem}.pdf"
                    drawing_hash = pdf_content_sha256(pdf)
                    comparison = compare_embedded(args.submission_pdf.resolve(), output["page_index"],
                                                  output["xobject_name"], pdf)
                    checked = {
                        "output_stem": stem, "file_checks": checks,
                        "drawing_sha256": drawing_hash,
                        "expected_drawing_sha256": output["drawing_sha256"],
                        "drawing_matches": drawing_hash == output["drawing_sha256"],
                        "submission_comparison": comparison,
                    }
                    checked["verified"] = bool(
                        all(item["verified"] or item["verification_status"] == "no_original"
                            for item in checks) and checked["drawing_matches"]
                        and comparison["verified_match"])
                    record["outputs"].append(checked)
                record["verified"] = all(item["verified"] for item in record["outputs"])
                if not record["verified"]:
                    record["error"] = "Full-file, drawing-stream, or submitted-resource comparison failed"
                print(f"Figure {number}: {'PDF EXACT MATCH' if record['verified'] else 'FAILED'}", flush=True)
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
            finally:
                record["completed_utc"] = _utc()
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipt["requested_coverage_complete"] = (
            len(receipt["figures"]) == len(requested)
            and all(item["verified"] for item in receipt["figures"]))
        receipt["all_supported_figures_verified"] = (
            receipt["requested_all_supported_figures"] and receipt["requested_coverage_complete"])
        receipt["status"] = "verified" if receipt["requested_coverage_complete"] else "failed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
    finally:
        receipt["completed_utc"] = _utc()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Verification receipt: {receipt_path}", flush=True)
    if receipt["status"] != "verified":
        print(receipt.get("error", "At least one figure failed exact reproduction"), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--submission-pdf", type=Path, required=True)
    parser.add_argument("--legacy-python", default=sys.executable)
    parser.add_argument("--modern-python", required=True)
    parser.add_argument("--figure2-python",
                        help="Required when Figure 2 is selected (separate frozen environment)")
    parser.add_argument("--figures", type=int, nargs="+", default=None,
                        help="Default: every figure declared in manifest.json")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"Refusing reproduction: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
