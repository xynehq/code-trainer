import os
import sys
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HackageDownloader:
    def __init__(self, packages_file: str, output_dir: str):
        self.packages_file = Path(packages_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read packages
        with open(self.packages_file, 'r') as f:
            self.packages = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(self.packages)} packages to download")
    
    def download_package(self, package: str) -> tuple[str, bool, str]:
        """
        Download a single package from Hackage
        Returns: (package_name, success, message)
        """
        try:
            # Parse package-version
            parts = package.rsplit('-', 1)
            if len(parts) != 2:
                return (package, False, "Invalid format")
            
            pkg_name, version = parts
            
            # Skip GHC internal packages (they come with GHC)
            ghc_internal = [
                'array', 'base', 'binary', 'bytestring', 'Cabal', 
                'containers', 'deepseq', 'directory', 'filepath', 
                'ghc', 'ghc-bignum', 'ghc-boot', 'ghc-boot-th', 
                'ghc-compact', 'ghc-heap', 'ghc-prim', 'ghci',
                'haskeline', 'hpc', 'integer-gmp', 'libiserv',
                'mtl', 'parsec', 'pretty', 'process', 'rts',
                'stm', 'template-haskell', 'terminfo', 'text',
                'time', 'transformers', 'unix', 'xhtml'
            ]
            
            if pkg_name in ghc_internal:
                return (package, True, "GHC internal package (skipped)")
            
            # Create package directory
            pkg_dir = self.output_dir / pkg_name / version
            
            # Check if already downloaded
            if pkg_dir.exists() and any(pkg_dir.glob("*.hs")):
                return (package, True, "Already exists")
            
            pkg_dir.mkdir(parents=True, exist_ok=True)
            
            # Download tarball from Hackage
            url = f"https://hackage.haskell.org/package/{pkg_name}-{version}/{pkg_name}-{version}.tar.gz"
            tarball = pkg_dir.parent / f"{pkg_name}-{version}.tar.gz"
            
            # Use curl to download
            result = subprocess.run(
                ['curl', '-L', '-f', '-o', str(tarball), url],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return (package, False, "Download failed")
            
            # Extract tarball
            result = subprocess.run(
                ['tar', '-xzf', str(tarball), '-C', str(pkg_dir.parent)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return (package, False, "Extraction failed")
            
            # Remove tarball to save space
            tarball.unlink()
            
            return (package, True, "Downloaded successfully")
            
        except subprocess.TimeoutExpired:
            return (package, False, "Timeout")
        except Exception as e:
            return (package, False, f"Error: {str(e)}")
    
    def download_all(self, max_workers: int = 10):
        """Download all packages concurrently"""
        logger.info(f"Starting download with {max_workers} workers...")
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_packages': []
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_pkg = {
                executor.submit(self.download_package, pkg): pkg 
                for pkg in self.packages
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(self.packages), desc="Downloading", unit="pkg") as pbar:
                for future in as_completed(future_to_pkg):
                    package, success, message = future.result()
                    
                    if success:
                        if "skipped" in message.lower() or "exists" in message.lower():
                            results['skipped'] += 1
                        else:
                            results['success'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_packages'].append((package, message))
                    
                    pbar.set_postfix({
                        'OK': results['success'],
                        'Skip': results['skipped'],
                        'Fail': results['failed']
                    })
                    pbar.update(1)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("Download Summary:")
        logger.info("=" * 70)
        logger.info(f"  Successfully downloaded: {results['success']}")
        logger.info(f"  Skipped (existing/GHC): {results['skipped']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info("=" * 70)
        
        if results['failed_packages']:
            logger.warning("\nFailed packages:")
            for pkg, msg in results['failed_packages'][:10]:  # Show first 10
                logger.warning(f"  - {pkg}: {msg}")
            if len(results['failed_packages']) > 10:
                logger.warning(f"  ... and {len(results['failed_packages']) - 10} more")
        
        # Save failed packages list
        if results['failed_packages']:
            failed_file = self.output_dir.parent / "failed_packages.txt"
            with open(failed_file, 'w') as f:
                for pkg, msg in results['failed_packages']:
                    f.write(f"{pkg}\t{msg}\n")
            logger.info(f"\nFailed packages list saved to: {failed_file}")
        
        return results


def main():
    downloader = HackageDownloader(
        packages_file="packages.txt",
        output_dir="HS_packages"
    )
    
    results = downloader.download_all(max_workers=10)
    
    if results['success'] > 0 or results['skipped'] > 0:
        logger.info("\n✓ Download completed! You can now run prepare_dataset.py")
    else:
        logger.error("\n✗ All downloads failed. Please check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
