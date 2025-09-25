#!/usr/bin/env python3
# -------------------------------------------------------------------------
# File: update_chunk_page_properties.py
# Project: APH-IF
# Author: APH-IF Development Team
# Date: 2025-08-25
# File Path: data_processing/update_chunk_page_properties.py
# -------------------------------------------------------------------------

# --- Module Objective ---
#   Update existing chunk nodes in Neo4j to have enhanced page tracking
#   properties while maintaining backward compatibility with the existing
#   'page' property. This enables better document navigation and retrieval.
# -------------------------------------------------------------------------

"""
Update Chunk Page Properties in Neo4j Knowledge Graph

Purpose
-------
Enhance existing chunk nodes with additional page tracking properties:
- page_start: First page where chunk begins (mirrors existing 'page')
- page_end: Last page where chunk ends (initially same as page_start)
- pages: Array of all pages touched (initially single-element array)
- page_span: Number of pages spanned (initially 1)

How it works
------------
1. Connects to Neo4j using environment variables
2. Updates all chunk nodes with new properties
3. Maintains backward compatibility with existing 'page' property
4. Provides verification of the updates

Environment variables
---------------------
- NEO4J_URI: Neo4j database URI
- NEO4J_USERNAME: Database username  
- NEO4J_PASSWORD: Database password
"""

# =========================================================================
# Imports
# =========================================================================
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
import argparse
import time

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =========================================================================
# Class Definitions
# =========================================================================

class ChunkPageUpdater:
    """Updates chunk nodes with enhanced page tracking properties."""
    
    def __init__(self, dry_run: bool = False):
        """Initialize connection to Neo4j database.
        
        Args:
            dry_run: If True, only preview changes without applying them
        """
        self.dry_run = dry_run
        
        # Get Neo4j credentials from environment
        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Missing Neo4j credentials in environment")
        
        print(f"Connecting to Neo4j at {self.uri}...")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        print("✅ Connected to Neo4j")
        
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get statistics about chunks in the database."""
        with self.driver.session() as session:
            # Total chunks
            result = session.run("MATCH (c:Chunk) RETURN count(c) as total")
            total_chunks = result.single()['total']
            
            # Chunks by document
            result = session.run("""
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.doc_id as doc_id, count(c) as chunk_count
                ORDER BY d.doc_id
            """)
            docs = {record['doc_id']: record['chunk_count'] for record in result}
            
            # Check if new properties already exist
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.page_start IS NOT NULL
                RETURN count(c) as updated_count
            """)
            updated_count = result.single()['updated_count']
            
            return {
                'total_chunks': total_chunks,
                'documents': docs,
                'already_updated': updated_count
            }
    
    def update_page_properties(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Update chunk nodes with enhanced page tracking properties.
        
        Args:
            batch_size: Number of chunks to update per transaction
            
        Returns:
            Dictionary with update statistics
        """
        print("\n" + "="*60)
        print("UPDATING CHUNK PAGE PROPERTIES")
        print("="*60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        start_time = time.time()
        stats = {'chunks_updated': 0, 'errors': 0, 'skipped': 0}
        
        with self.driver.session() as session:
            # Get total chunks to process
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.page IS NOT NULL AND c.page_start IS NULL
                RETURN count(c) as total
            """)
            total_to_update = result.single()['total']
            
            if total_to_update == 0:
                print("✅ All chunks already have enhanced page properties")
                return stats
            
            print(f"Found {total_to_update:,} chunks to update")
            
            if not self.dry_run:
                # Process in batches for better performance
                processed = 0
                
                while processed < total_to_update:
                    # Update batch of chunks
                    result = session.run("""
                        MATCH (c:Chunk)
                        WHERE c.page IS NOT NULL AND c.page_start IS NULL
                        WITH c LIMIT $batch_size
                        SET c.page_start = c.page,
                            c.page_end = c.page,
                            c.pages = [c.page],
                            c.page_span = 1
                        RETURN count(c) as updated
                    """, batch_size=batch_size)
                    
                    batch_updated = result.single()['updated']
                    processed += batch_updated
                    stats['chunks_updated'] += batch_updated
                    
                    # Progress reporting
                    progress = (processed / total_to_update) * 100
                    print(f"Progress: {processed:,}/{total_to_update:,} ({progress:.1f}%)")
                    
                    if batch_updated < batch_size:
                        break  # No more chunks to process
            else:
                # Dry run - just show what would be updated
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.page IS NOT NULL AND c.page_start IS NULL
                    RETURN c.chunk_id as chunk_id, c.page as page
                    LIMIT 10
                """)
                
                print("\nSample chunks that would be updated:")
                for record in result:
                    print(f"  {record['chunk_id']}: page {record['page']} → "
                          f"page_start={record['page']}, page_end={record['page']}, "
                          f"pages=[{record['page']}], page_span=1")
                
                stats['chunks_updated'] = total_to_update
        
        elapsed = time.time() - start_time
        stats['elapsed_seconds'] = elapsed
        
        return stats
    
    def verify_updates(self, sample_size: int = 10) -> None:
        """Verify that updates were applied correctly.
        
        Args:
            sample_size: Number of chunks to sample for verification
        """
        print("\n" + "="*60)
        print("VERIFYING UPDATES")
        print("="*60)
        
        with self.driver.session() as session:
            # Check chunks with new properties
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.page_start IS NOT NULL
                RETURN c.chunk_id as chunk_id,
                       c.page as page,
                       c.page_start as page_start,
                       c.page_end as page_end,
                       c.pages as pages,
                       c.page_span as page_span
                LIMIT $limit
            """, limit=sample_size)
            
            print(f"\nSample of updated chunks:")
            for record in result:
                print(f"\n  Chunk: {record['chunk_id']}")
                print(f"    Original page: {record['page']}")
                print(f"    page_start: {record['page_start']}")
                print(f"    page_end: {record['page_end']}")
                print(f"    pages: {record['pages']}")
                print(f"    page_span: {record['page_span']}")
            
            # Summary statistics
            result = session.run("""
                MATCH (c:Chunk)
                RETURN 
                    count(c) as total,
                    count(c.page_start) as with_page_start,
                    count(c.page_end) as with_page_end,
                    count(c.pages) as with_pages,
                    count(c.page_span) as with_page_span
            """)
            
            stats = result.single()
            print(f"\n📊 Update Statistics:")
            print(f"  Total chunks: {stats['total']:,}")
            print(f"  With page_start: {stats['with_page_start']:,}")
            print(f"  With page_end: {stats['with_page_end']:,}")
            print(f"  With pages array: {stats['with_pages']:,}")
            print(f"  With page_span: {stats['with_page_span']:,}")
            
            # Check consistency
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.page_start IS NOT NULL AND c.page <> c.page_start
                RETURN count(c) as inconsistent
            """)
            
            inconsistent = result.single()['inconsistent']
            if inconsistent > 0:
                print(f"\n⚠️  Warning: {inconsistent} chunks have page != page_start")
            else:
                print(f"\n✅ All chunks have consistent page properties")

# =========================================================================
# Standalone Function Definitions
# =========================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update chunk nodes with enhanced page tracking properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview changes without applying
    python update_chunk_page_properties.py --dry-run
    
    # Apply updates with default batch size
    python update_chunk_page_properties.py
    
    # Apply updates with custom batch size
    python update_chunk_page_properties.py --batch-size 500
    
    # Skip verification after update
    python update_chunk_page_properties.py --no-verify
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without applying them')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of chunks to update per transaction')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification after update')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("APH-IF Chunk Page Property Updater")
    print("="*60)
    
    updater = None
    try:
        updater = ChunkPageUpdater(dry_run=args.dry_run)
        
        # Show current statistics
        print("\n📊 Current Database Statistics:")
        stats = updater.get_chunk_statistics()
        print(f"  Total chunks: {stats['total_chunks']:,}")
        print(f"  Documents: {len(stats['documents'])}")
        for doc_id, count in stats['documents'].items():
            print(f"    - {doc_id}: {count:,} chunks")
        
        if stats['already_updated'] > 0:
            print(f"\n  ℹ️  {stats['already_updated']:,} chunks already have enhanced properties")
        
        # Perform update
        update_stats = updater.update_page_properties(batch_size=args.batch_size)
        
        # Show results
        print("\n" + "="*60)
        print("UPDATE COMPLETE")
        print("="*60)
        
        if args.dry_run:
            print(f"🔍 Dry run complete - {update_stats['chunks_updated']:,} chunks would be updated")
        else:
            print(f"✅ Successfully updated {update_stats['chunks_updated']:,} chunks")
            if 'elapsed_seconds' in update_stats:
                print(f"⏱️  Time elapsed: {update_stats['elapsed_seconds']:.2f} seconds")
        
        # Verify updates
        if not args.no_verify and not args.dry_run:
            updater.verify_updates()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if updater:
            updater.close()
    
    return 0

# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
if __name__ == "__main__":
    sys.exit(main())