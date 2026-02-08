#!/usr/bin/env python3
# File: scripts/monitor_resources.py
"""
Resource monitoring script for the fraud detection pipeline
Monitors memory, CPU, and database sizes in real-time
"""

import psutil
import time
import os
import argparse
from pathlib import Path
from datetime import datetime


def get_process_info(process_name='python'):
    """Get information about running Python processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('main_optimized' in cmd for cmd in cmdline):
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_directory_size(path):
    """Get total size of a directory"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except Exception:
        pass
    return total


def monitor_system(interval=5, duration=None):
    """
    Monitor system resources
    
    Args:
        interval: Seconds between measurements
        duration: Total monitoring duration in seconds (None for infinite)
    """
    print("=" * 80)
    print("FRAUD DETECTION PIPELINE - RESOURCE MONITOR")
    print("=" * 80)
    print(f"Monitoring interval: {interval} seconds")
    print(f"Press Ctrl+C to stop\n")
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    vector_db_dir = base_dir / "databases" / "vector_store"
    
    start_time = time.time()
    measurement_count = 0
    
    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break
            
            measurement_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process-specific metrics
            processes = get_process_info()
            
            print(f"\n[{timestamp}] Measurement #{measurement_count}")
            print("-" * 80)
            
            # System metrics
            print(f"System CPU: {cpu_percent}%")
            print(f"System RAM: {format_bytes(memory.used)} / {format_bytes(memory.total)} "
                  f"({memory.percent}%)")
            print(f"System Swap: {format_bytes(swap.used)} / {format_bytes(swap.total)} "
                  f"({swap.percent}%)")
            
            # Process metrics
            if processes:
                print(f"\nPipeline Processes ({len(processes)}):")
                total_process_memory = 0
                for proc in processes:
                    try:
                        proc_mem = proc.memory_info()
                        proc_cpu = proc.cpu_percent()
                        total_process_memory += proc_mem.rss
                        
                        print(f"  PID {proc.pid}:")
                        print(f"    Memory (RSS): {format_bytes(proc_mem.rss)}")
                        print(f"    CPU: {proc_cpu}%")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                print(f"  Total Process Memory: {format_bytes(total_process_memory)}")
            else:
                print("\nNo pipeline processes found")
            
            # Database sizes
            print("\nDatabase Sizes:")
            if vector_db_dir.exists():
                vdb_size = get_directory_size(vector_db_dir)
                print(f"  Vector DB: {format_bytes(vdb_size)}")
            else:
                print("  Vector DB: Not found")
            
            # Docker container stats (if available)
            try:
                import docker
                client = docker.from_env()
                neo4j_container = client.containers.get('neo4j-fraud-detection')
                stats = neo4j_container.stats(stream=False)
                
                mem_usage = stats['memory_stats'].get('usage', 0)
                mem_limit = stats['memory_stats'].get('limit', 0)
                
                print(f"  Neo4j Container: {format_bytes(mem_usage)} / {format_bytes(mem_limit)}")
            except Exception:
                print("  Neo4j Container: Unable to get stats")
            
            # Wait for next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(f"Total measurements: {measurement_count}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor resource usage of fraud detection pipeline'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Seconds between measurements (default: 5)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Total monitoring duration in seconds (default: infinite)'
    )
    
    args = parser.parse_args()
    
    monitor_system(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()