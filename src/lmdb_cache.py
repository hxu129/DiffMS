"""
Shared cache implementation using LMDB for memory-efficient multi-process caching.

Benefits:
- Multiple processes share the same physical memory (via mmap)
- Fast read/write performance
- Automatic persistence to disk
- Concurrent access support
"""

import lmdb
import pickle
import logging
import hashlib
from typing import Optional, Any
import numpy as np


class LMDBCache:
    """
    LMDB-based cache for spectrum and score data.
    
    Memory efficient: All processes share the same memory via mmap.
    Thread/Process safe: LMDB handles concurrent access automatically.
    
    Example:
        cache = LMDBCache(cache_dir='./cache', map_size=10*1024**3)  # 10GB max
        
        # Write
        cache.put_spectrum('SMILES', '[M+H]+', spectrum_array)
        cache.put_score('SMILES', '[M+H]+', target_hash, bin_size, 0.85)
        
        # Read
        spectrum = cache.get_spectrum('SMILES', '[M+H]+')
        score = cache.get_score('SMILES', '[M+H]+', target_hash, bin_size)
    """
    
    def __init__(self, cache_dir: str, map_size: int = 10 * 1024**3):
        """
        Initialize LMDB cache.
        
        Args:
            cache_dir: Directory to store LMDB database
            map_size: Maximum size of database in bytes (default: 10GB)
                     This is virtual memory, not actual memory usage.
        """
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_dir = cache_dir
        self.map_size = map_size
        
        # Open LMDB environment
        # writemap=True allows multiple processes to write
        # map_async=True improves write performance
        # lock=True ensures proper concurrent access
        self.env = lmdb.open(
            cache_dir,
            map_size=map_size,
            max_dbs=2,  # Two databases: spectra and scores
            writemap=True,
            map_async=True,
            lock=True,
            readahead=True,
            meminit=False,
            max_readers=126  # Allow many concurrent readers
        )
        
        # Open two sub-databases
        self.spectra_db = self.env.open_db(b'spectra')
        self.scores_db = self.env.open_db(b'scores')
        
        logging.info(f"Initialized LMDB cache at {cache_dir} (max_size={map_size/(1024**3):.1f}GB)")
        self._log_stats()
    
    def _make_spectrum_key(self, smiles: str, adduct: str) -> bytes:
        """Create key for spectrum cache."""
        key = f"{smiles}|{adduct}"
        return key.encode('utf-8')
    
    def _make_score_key(self, smiles: str, adduct: str, target_hash: str, bin_size: float) -> bytes:
        """Create key for score cache."""
        key = f"{smiles}|{adduct}|{target_hash}|{bin_size}"
        return key.encode('utf-8')
    
    def put_spectrum(self, smiles: str, adduct: str, spectrum: np.ndarray) -> bool:
        """
        Store spectrum in cache.
        
        Args:
            smiles: SMILES string
            adduct: Adduct string (e.g., '[M+H]+')
            spectrum: Numpy array of shape (n_peaks, 2) with [m/z, intensity]
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_spectrum_key(smiles, adduct)
            value = pickle.dumps(spectrum, protocol=4)
            
            with self.env.begin(db=self.spectra_db, write=True) as txn:
                txn.put(key, value)
            return True
        except Exception as e:
            logging.error(f"Failed to store spectrum: {e}")
            return False
    
    def get_spectrum(self, smiles: str, adduct: str) -> Optional[np.ndarray]:
        """
        Retrieve spectrum from cache.
        
        Returns:
            Spectrum array if found, None otherwise
        """
        try:
            key = self._make_spectrum_key(smiles, adduct)
            
            with self.env.begin(db=self.spectra_db, write=False) as txn:
                value = txn.get(key)
                if value is None:
                    return None
                return pickle.loads(value)
        except Exception as e:
            logging.error(f"Failed to retrieve spectrum: {e}")
            return None
    
    def put_score(self, smiles: str, adduct: str, target_hash: str, 
                  bin_size: float, score: float) -> bool:
        """
        Store similarity score in cache.
        
        Args:
            smiles: SMILES string
            adduct: Adduct string
            target_hash: Hash of target spectrum
            bin_size: Bin size used for scoring
            score: Similarity score (float)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_score_key(smiles, adduct, target_hash, bin_size)
            value = pickle.dumps(score, protocol=4)
            
            with self.env.begin(db=self.scores_db, write=True) as txn:
                txn.put(key, value)
            return True
        except Exception as e:
            logging.error(f"Failed to store score: {e}")
            return False
    
    def get_score(self, smiles: str, adduct: str, target_hash: str, 
                  bin_size: float) -> Optional[float]:
        """
        Retrieve similarity score from cache.
        
        Returns:
            Score if found, None otherwise
        """
        try:
            key = self._make_score_key(smiles, adduct, target_hash, bin_size)
            
            with self.env.begin(db=self.scores_db, write=False) as txn:
                value = txn.get(key)
                if value is None:
                    return None
                return pickle.loads(value)
        except Exception as e:
            logging.error(f"Failed to retrieve score: {e}")
            return None
    
    def batch_get_spectra(self, keys: list) -> dict:
        """
        Batch retrieve multiple spectra.
        
        Args:
            keys: List of (smiles, adduct) tuples
        
        Returns:
            Dictionary mapping keys to spectra (only for found items)
        """
        result = {}
        try:
            with self.env.begin(db=self.spectra_db, write=False) as txn:
                for smiles, adduct in keys:
                    key = self._make_spectrum_key(smiles, adduct)
                    value = txn.get(key)
                    if value is not None:
                        result[(smiles, adduct)] = pickle.loads(value)
        except Exception as e:
            logging.error(f"Batch spectrum retrieval failed: {e}")
        return result
    
    def batch_get_scores(self, keys: list) -> dict:
        """
        Batch retrieve multiple scores.
        
        Args:
            keys: List of (smiles, adduct, target_hash, bin_size) tuples
        
        Returns:
            Dictionary mapping keys to scores (only for found items)
        """
        result = {}
        try:
            with self.env.begin(db=self.scores_db, write=False) as txn:
                for smiles, adduct, target_hash, bin_size in keys:
                    key = self._make_score_key(smiles, adduct, target_hash, bin_size)
                    value = txn.get(key)
                    if value is not None:
                        result[(smiles, adduct, target_hash, bin_size)] = pickle.loads(value)
        except Exception as e:
            logging.error(f"Batch score retrieval failed: {e}")
        return result
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.env.stat()
        info = self.env.info()
        
        # Count entries in each database
        with self.env.begin(db=self.spectra_db, write=False) as txn:
            spectra_count = txn.stat()['entries']
        
        with self.env.begin(db=self.scores_db, write=False) as txn:
            scores_count = txn.stat()['entries']
        
        return {
            'spectra_count': spectra_count,
            'scores_count': scores_count,
            'total_entries': spectra_count + scores_count,
            'db_size_mb': info['map_size'] / (1024**2),
            'used_size_mb': (stats['psize'] * info['last_pgno']) / (1024**2),
            'cache_dir': self.cache_dir
        }
    
    def _log_stats(self):
        """Log cache statistics."""
        stats = self.get_stats()
        logging.info(
            f"LMDB Cache Stats: {stats['spectra_count']} spectra, "
            f"{stats['scores_count']} scores, "
            f"using {stats['used_size_mb']:.1f}MB / {stats['db_size_mb']:.0f}MB"
        )
    
    def sync(self):
        """Force sync to disk."""
        self.env.sync()
    
    def close(self):
        """Close the database."""
        self.env.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.env.close()
        except:
            pass


def migrate_pickle_to_lmdb(pickle_dir: str, lmdb_dir: str):
    """
    Migrate existing pickle caches to LMDB format.
    
    Args:
        pickle_dir: Directory containing spectra_cache_*.pkl files
        lmdb_dir: Directory for LMDB database
    """
    import os
    import glob
    
    logging.info(f"Migrating pickle caches from {pickle_dir} to LMDB at {lmdb_dir}")
    
    # Create LMDB cache
    lmdb_cache = LMDBCache(lmdb_dir, map_size=20 * 1024**3)  # 20GB max
    
    # Find all pickle files
    spectra_files = glob.glob(os.path.join(pickle_dir, 'spectra_cache_*.pkl'))
    scores_files = glob.glob(os.path.join(pickle_dir, 'scores_cache_*.pkl'))
    
    # Migrate spectra
    total_spectra = 0
    for filepath in spectra_files:
        logging.info(f"Migrating {os.path.basename(filepath)}...")
        try:
            with open(filepath, 'rb') as f:
                cache = pickle.load(f)
            
            for (smiles, adduct), spectrum in cache.items():
                lmdb_cache.put_spectrum(smiles, adduct, spectrum)
                total_spectra += 1
                
                if total_spectra % 10000 == 0:
                    logging.info(f"Migrated {total_spectra} spectra...")
        except Exception as e:
            logging.error(f"Failed to migrate {filepath}: {e}")
    
    # Migrate scores
    total_scores = 0
    for filepath in scores_files:
        logging.info(f"Migrating {os.path.basename(filepath)}...")
        try:
            with open(filepath, 'rb') as f:
                cache = pickle.load(f)
            
            for (smiles, adduct, target_hash, bin_size), score in cache.items():
                lmdb_cache.put_score(smiles, adduct, target_hash, bin_size, score)
                total_scores += 1
                
                if total_scores % 10000 == 0:
                    logging.info(f"Migrated {total_scores} scores...")
        except Exception as e:
            logging.error(f"Failed to migrate {filepath}: {e}")
    
    lmdb_cache.sync()
    lmdb_cache._log_stats()
    
    logging.info(f"Migration complete: {total_spectra} spectra, {total_scores} scores")
    return lmdb_cache


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    # Create cache
    cache = LMDBCache('./test_cache', map_size=1024**3)
    
    # Test spectrum storage
    test_spectrum = np.array([[100.0, 0.5], [200.0, 1.0], [300.0, 0.3]])
    cache.put_spectrum('CCO', '[M+H]+', test_spectrum)
    
    # Test retrieval
    retrieved = cache.get_spectrum('CCO', '[M+H]+')
    assert np.allclose(retrieved, test_spectrum)
    
    # Test score storage
    cache.put_score('CCO', '[M+H]+', 'hash123', 20.0, 0.85)
    score = cache.get_score('CCO', '[M+H]+', 'hash123', 20.0)
    assert score == 0.85
    
    print("All tests passed!")
    cache._log_stats()

