"""
HPC Maturity Assessment Agent - Konfiguracija
===============================================
Definicije konstanti, pragova i parametara za procjenu zrelosti.
"""

# Nivoi zrelosti (Maturity Levels)
MATURITY_LEVELS = {
    1: {"name": "Početni", "description": "Minimalna HPC spremnost, potrebna značajna ulaganja"},
    2: {"name": "Razvijajući", "description": "Osnovni kapaciteti postoje, potrebna obuka i optimizacija"},
    3: {"name": "Definisani", "description": "Strukturirani procesi, djelimična paralelizacija"},
    4: {"name": "Upravljani", "description": "Optimizovani procesi, efikasno korištenje resursa"},
    5: {"name": "Optimizovani", "description": "Potpuna HPC spremnost, kontinuirano unapređenje"}
}

# Pragovi za maturity score
MATURITY_THRESHOLDS = {
    1: (0, 20),
    2: (20, 40),
    3: (40, 60),
    4: (60, 80),
    5: (80, 100)
}

# Težinski faktori za komponente procjene
ASSESSMENT_WEIGHTS = {
    "infrastructure": 0.25,
    "team_readiness": 0.25,
    "software_optimization": 0.25,
    "workload_characteristics": 0.25
}

# Parametri za simulaciju
SIMULATION_PARAMS = {
    "cpu_cost_per_hour": 0.05,      # EUR po CPU-satu
    "gpu_cost_per_hour": 0.50,      # EUR po GPU-satu
    "memory_cost_per_gb_hour": 0.01, # EUR po GB-satu
    "storage_cost_per_tb_month": 20, # EUR po TB mjesečno
    "network_cost_per_gb": 0.02,    # EUR po GB transferiranog
    "efficiency_baseline": 0.60,    # Bazna efikasnost bez optimizacije
    "parallel_overhead": 0.10       # Overhead za paralelizaciju
}

# HPC Benchmark reference vrijednosti
HPC_BENCHMARKS = {
    "linpack": {
        "low": 100,      # GFLOPS - niska performansa
        "medium": 500,   # GFLOPS - srednja
        "high": 2000     # GFLOPS - visoka
    },
    "memory_bandwidth": {
        "low": 20,       # GB/s
        "medium": 100,
        "high": 400
    },
    "interconnect_latency": {
        "low": 10,       # μs (niže je bolje)
        "medium": 2,
        "high": 0.5
    }
}

# Tipovi workload-a i njihove karakteristike
WORKLOAD_TYPES = {
    "compute_intensive": {
        "cpu_weight": 0.8,
        "memory_weight": 0.1,
        "io_weight": 0.1,
        "parallelization_potential": 0.9
    },
    "memory_intensive": {
        "cpu_weight": 0.3,
        "memory_weight": 0.6,
        "io_weight": 0.1,
        "parallelization_potential": 0.7
    },
    "io_intensive": {
        "cpu_weight": 0.2,
        "memory_weight": 0.2,
        "io_weight": 0.6,
        "parallelization_potential": 0.5
    },
    "balanced": {
        "cpu_weight": 0.4,
        "memory_weight": 0.3,
        "io_weight": 0.3,
        "parallelization_potential": 0.7
    },
    "gpu_accelerated": {
        "cpu_weight": 0.2,
        "memory_weight": 0.2,
        "io_weight": 0.1,
        "gpu_weight": 0.5,
        "parallelization_potential": 0.95
    }
}

# Kategorije vještina za evaluaciju tima
TEAM_SKILL_CATEGORIES = {
    "parallel_programming": {
        "weight": 0.30,
        "skills": ["MPI", "OpenMP", "CUDA", "threading", "async_programming"]
    },
    "hpc_tools": {
        "weight": 0.20,
        "skills": ["slurm", "pbs", "profilers", "debuggers", "containers"]
    },
    "optimization": {
        "weight": 0.25,
        "skills": ["vectorization", "memory_optimization", "cache_optimization", "algorithm_complexity"]
    },
    "devops": {
        "weight": 0.15,
        "skills": ["version_control", "ci_cd", "automation", "monitoring"]
    },
    "domain_knowledge": {
        "weight": 0.10,
        "skills": ["numerical_methods", "scientific_computing", "data_analysis"]
    }
}

# Software anti-patterns koji ukazuju na probleme
SOFTWARE_ANTIPATTERNS = {
    "sequential_loops": {"penalty": 15, "description": "Sekvencijalne petlje bez paralelizacije"},
    "global_variables": {"penalty": 10, "description": "Prekomjerno korištenje globalnih varijabli"},
    "memory_leaks": {"penalty": 20, "description": "Potencijalni memory leaks"},
    "blocking_io": {"penalty": 12, "description": "Blokirajuće I/O operacije"},
    "poor_data_locality": {"penalty": 15, "description": "Loša lokalnost podataka"},
    "excessive_synchronization": {"penalty": 18, "description": "Previše sinhronizacionih tačaka"},
    "unoptimized_algorithms": {"penalty": 20, "description": "Neoptimizovani algoritmi (O(n²) umjesto O(n log n))"}
}

# Pozitivni patterns
SOFTWARE_PATTERNS = {
    "vectorized_operations": {"bonus": 15, "description": "Vektorizovane operacije"},
    "parallel_loops": {"bonus": 20, "description": "Paralelne petlje (OpenMP/MPI)"},
    "async_io": {"bonus": 10, "description": "Asinhroni I/O"},
    "memory_pooling": {"bonus": 12, "description": "Memory pooling"},
    "cache_friendly": {"bonus": 15, "description": "Cache-friendly pristup podacima"},
    "gpu_offloading": {"bonus": 25, "description": "GPU offloading"}
}
