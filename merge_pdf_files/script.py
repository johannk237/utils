#!/usr/bin/env python3
"""
Pipeline industriel:
1) Réparer chaque PDF avec pikepdf
2) Compresser chaque PDF (par fichier) en parallèle via qpdf CLI
3) Fusionner en plusieurs fichiers de max 200Mo chacun
4) Compression finale de chaque fichier fusionné via qpdf

Usage:
    python script.py --workers 6 --keep-temp --max-size 200
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time
import logging

try:
    from pikepdf import Pdf, PdfError
except Exception as e:
    print("Erreur: pikepdf est requis. Installe avec `pip install pikepdf`.")
    raise e

# -------------------------
# Configuration & Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -------------------------
# Helpers
# -------------------------
def safe_name(path: str) -> str:
    """Return a safe basename for outputs."""
    return Path(path).name

def run_cmd(cmd, timeout=None):
    """Run subprocess command, return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        return proc.returncode, proc.stdout.decode(errors='ignore'), proc.stderr.decode(errors='ignore')
    except Exception as e:
        return 2, "", str(e)

# -------------------------
# PDF operations
# -------------------------
def repair_with_pikepdf(input_file: str, repaired_file: str) -> bool:
    """Open and save with pikepdf to repair / normalize."""
    try:
        with Pdf.open(input_file) as pdf:
            # save with linearize True for better structure
            pdf.save(repaired_file, linearize=True)
        logging.debug(f"Repaired {input_file} -> {repaired_file}")
        return True
    except PdfError as e:
        logging.debug(f"pikepdf failed to repair {input_file}: {e}")
        return False
    except Exception as e:
        logging.debug(f"Unexpected error repairing {input_file}: {e}")
        return False

def compress_with_qpdf(input_file: str, output_file: str, extra_flags=None) -> bool:
    """
    Compress / rewrite structure with qpdf.
    Uses object stream generation + linearize by default.
    """
    if extra_flags is None:
        extra_flags = []

    cmd = ["qpdf", "--object-streams=generate", "--linearize"] + extra_flags + [input_file, output_file]
    rc, out, err = run_cmd(cmd, timeout=600)
    if rc == 0:
        logging.debug(f"qpdf compressed {input_file} -> {output_file}")
        return True
    else:
        logging.debug(f"qpdf failed for {input_file}: rc={rc}, err={err[:200]}")
        return False

def process_single_file(args):
    """
    Per-file pipeline (run in separate process):
    - repair -> compressed version (via qpdf)
    Returns tuple: (input_file, output_path or None, status_str)
    """
    input_file, tmp_dir, use_ghostscript = args
    basename = safe_name(input_file)
    repaired = os.path.join(tmp_dir, f"repaired_{basename}")
    compressed = os.path.join(tmp_dir, f"opt_{basename}")

    # Step 1: repair (pikepdf)
    ok = repair_with_pikepdf(input_file, repaired)
    if not ok:
        logging.warning(f"❌ Repair failed: {input_file}")
        return (input_file, None, "repair_failed")

    # Step 2: compress with qpdf
    ok = compress_with_qpdf(repaired, compressed)
    if not ok:
        logging.warning(f"⚠ qpdf compression failed for {repaired}. Trying Ghostscript fallback (if enabled).")
        # optional fallback with Ghostscript for image-heavy PDFs
        if use_ghostscript:
            gs_out = os.path.join(tmp_dir, f"gs_{basename}")
            gs_cmd = [
                "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/ebook", "-dNOPAUSE", "-dBATCH", "-dQUIET",
                f"-sOutputFile={gs_out}", repaired
            ]
            rc, _, _ = run_cmd(gs_cmd, timeout=900)
            if rc == 0:
                # then try qpdf on gs_out
                ok2 = compress_with_qpdf(gs_out, compressed)
                if ok2:
                    logging.info(f"✅ Ghostscript + qpdf succeeded for {input_file}")
                    return (input_file, compressed, "ok_gs_qpdf")
                else:
                    logging.warning(f"❌ After GS, qpdf still failed for {input_file}")
                    return (input_file, None, "compress_failed_after_gs")
            else:
                logging.warning(f"❌ Ghostscript failed for {input_file}")
                return (input_file, None, "gs_failed")
        else:
            return (input_file, None, "qpdf_failed")
    else:
        # success
        return (input_file, compressed, "ok")

# -------------------------
# Merge (sequential) - avec limite de taille
# -------------------------
def merge_into_chunks(input_files, output_prefix, max_size_mb=200, tmp_dir="."):
    """
    Merge list of input PDF paths into multiple output files.
    Each output file is limited to max_size_mb (approximately).
    Returns list of output file paths.
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    output_files = []
    chunk_index = 1
    current_pdf = Pdf.new()
    current_size = 0
    total_pages = 0
    
    logging.info(f"🔗 Merging {len(input_files)} files into chunks of max {max_size_mb}MB ...")
    start = time()
    
    for i, p in enumerate(input_files):
        try:
            # Check file size
            file_size = os.path.getsize(p)
            
            # If adding this file would exceed the limit, save current chunk
            if current_size > 0 and (current_size + file_size) > max_size_bytes:
                # Save current chunk
                output_path = f"{output_prefix}_part{chunk_index:03d}.pdf"
                try:
                    current_pdf.save(output_path)
                    actual_size = os.path.getsize(output_path)
                    logging.info(f"✅ Saved chunk {chunk_index}: {output_path} ({actual_size/(1024*1024):.2f}MB, {len(current_pdf.pages)} pages)")
                    output_files.append(output_path)
                    current_pdf.close()
                except Exception as e:
                    logging.error(f"❌ Failed to save chunk {chunk_index}: {e}")
                    current_pdf.close()
                
                # Start new chunk
                chunk_index += 1
                current_pdf = Pdf.new()
                current_size = 0
            
            # Add pages from current file
            with Pdf.open(p) as src:
                current_pdf.pages.extend(src.pages)
                current_size += file_size
                total_pages += len(src.pages)
                logging.debug(f"Added {len(src.pages)} pages from {p} (file {i+1}/{len(input_files)})")
                
        except Exception as e:
            logging.warning(f"⚠ Skipping {p} during merge: {e}")
    
    # Save last chunk if it has pages
    if len(current_pdf.pages) > 0:
        output_path = f"{output_prefix}_part{chunk_index:03d}.pdf"
        try:
            current_pdf.save(output_path)
            actual_size = os.path.getsize(output_path)
            logging.info(f"✅ Saved chunk {chunk_index}: {output_path} ({actual_size/(1024*1024):.2f}MB, {len(current_pdf.pages)} pages)")
            output_files.append(output_path)
        except Exception as e:
            logging.error(f"❌ Failed to save final chunk: {e}")
        finally:
            current_pdf.close()
    else:
        current_pdf.close()
    
    elapsed = time() - start
    logging.info(f"✅ Merge done: {len(output_files)} chunks created ({total_pages} pages total) in {elapsed:.1f}s")
    return output_files

# -------------------------
# Main pipeline
# -------------------------
def main(folder=".", workers=None, keep_temp=False, final_prefix="MERGED", max_size_mb=200, use_ghostscript=False):
    start_time = time()
    folder = os.path.abspath(folder)
    os.chdir(folder)
    files = sorted([f for f in glob.glob("*.pdf") if not f.startswith(final_prefix)])
    if not files:
        logging.info("Aucun fichier PDF trouvé dans le dossier.")
        return

    if workers is None:
        workers = max(1, cpu_count() - 1)

    tmp_dir = os.path.join(folder, ".pdf_pipeline_tmp")
    if os.path.exists(tmp_dir) and not keep_temp:
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    logging.info(f"Found {len(files)} pdf(s). Workers: {workers}. Temp dir: {tmp_dir}")
    # Prepare args for pool
    pool_args = [(f, tmp_dir, use_ghostscript) for f in files]

    # Process in parallel (repair + compress per file)
    results = []
    with Pool(workers) as p:
        for res in p.imap_unordered(process_single_file, pool_args):
            input_file, out_path, status = res
            if status.startswith("ok"):
                logging.info(f"✅ {input_file} -> {os.path.basename(out_path)}")
            else:
                logging.warning(f"✖ {input_file} : {status}")
            results.append(res)

    # Collect successful compressed outputs in original file order
    compressed_map = {inp: out for (inp, out, st) in results if out}
    compressed_in_order = [compressed_map[f] for f in files if f in compressed_map]

    if not compressed_in_order:
        logging.error("Aucun fichier compressé valide pour fusion.")
        return

    # Merge into multiple chunks with size limit
    output_prefix = os.path.join(tmp_dir, final_prefix)
    merged_chunks = merge_into_chunks(compressed_in_order, output_prefix, max_size_mb=max_size_mb, tmp_dir=tmp_dir)
    
    if not merged_chunks:
        logging.error("Aucun fichier fusionné créé.")
        return

    # Final compression of each chunk with qpdf
    logging.info(f"📦 Running final compression on {len(merged_chunks)} chunk(s) with qpdf...")
    final_files = []
    for i, chunk_path in enumerate(merged_chunks):
        chunk_basename = os.path.basename(chunk_path)
        final_chunk_path = os.path.join(folder, f"COMPRESSED_{chunk_basename}")
        
        if compress_with_qpdf(chunk_path, final_chunk_path):
            actual_size = os.path.getsize(final_chunk_path)
            logging.info(f"🎉 Final compressed file {i+1}/{len(merged_chunks)}: {final_chunk_path} ({actual_size/(1024*1024):.2f}MB)")
            final_files.append(final_chunk_path)
        else:
            # If final compression fails, copy the uncompressed version
            logging.warning(f"⚠ Final qpdf compression failed for {chunk_path}; copying uncompressed version.")
            try:
                fallback_path = os.path.join(folder, chunk_basename)
                shutil.copyfile(chunk_path, fallback_path)
                logging.info(f"📌 Copied uncompressed file to {fallback_path}")
                final_files.append(fallback_path)
            except Exception as e:
                logging.error(f"❌ Failed to copy uncompressed file: {e}")

    # Cleanup
    if not keep_temp:
        try:
            shutil.rmtree(tmp_dir)
            logging.debug("Temporary directory removed.")
        except Exception:
            logging.debug("Could not remove temporary directory.")

    elapsed_total = time() - start_time
    logging.info(f"🎊 Pipeline terminé: {len(final_files)} fichier(s) créé(s) en {elapsed_total/60:.2f} minutes")
    for f in final_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logging.info(f"   📄 {os.path.basename(f)} ({size_mb:.2f}MB)")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline industriel de réparation, compression (par fichier) et fusion PDF en chunks de taille limitée.")
    ap.add_argument("--folder", "-d", default=".", help="Dossier contenant les PDFs (default: current dir).")
    ap.add_argument("--workers", "-w", type=int, default=None, help="Nombre de workers (default: cpu_count()-1).")
    ap.add_argument("--keep-temp", action="store_true", help="Conserver le dossier temporaire pour debug.")
    ap.add_argument("--final-prefix", default="MERGED", help="Préfixe des fichiers finaux fusionnés (default: MERGED).")
    ap.add_argument("--max-size", type=int, default=200, help="Taille maximale en Mo pour chaque fichier de sortie (default: 200).")
    ap.add_argument("--ghostscript", action="store_true", help="Activer fallback Ghostscript si qpdf échoue (utile pour images lourdes).")
    args = ap.parse_args()

    main(folder=args.folder, workers=args.workers, keep_temp=args.keep_temp, final_prefix=args.final_prefix, max_size_mb=args.max_size, use_ghostscript=args.ghostscript)