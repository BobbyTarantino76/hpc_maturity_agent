"""
Modul za procjenu optimizacije softvera
========================================
Analiza softvera/koda s ciljem procjene skalabilnosti,
paralelizacije i potencijalnih uskih grla.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
from enum import Enum

from config import SOFTWARE_ANTIPATTERNS, SOFTWARE_PATTERNS


class ParallelizationLevel(Enum):
    """Nivoi paralelizacije."""
    NONE = 0
    BASIC = 1
    MODERATE = 2
    ADVANCED = 3
    OPTIMAL = 4


@dataclass
class CodeMetrics:
    """Metrike analize koda."""
    total_lines: int = 0
    loop_count: int = 0
    parallel_loop_count: int = 0
    function_count: int = 0
    mpi_calls: int = 0
    openmp_pragmas: int = 0
    cuda_kernels: int = 0
    io_operations: int = 0
    memory_allocations: int = 0
    synchronization_points: int = 0


@dataclass
class SoftwareProfile:
    """Profil softverskog rješenja."""
    name: str
    language: str  # python, c, cpp, fortran
    source_code: Optional[str] = None
    estimated_loc: int = 0
    uses_mpi: bool = False
    uses_openmp: bool = False
    uses_cuda: bool = False
    uses_vectorization: bool = False
    has_profiling_data: bool = False
    measured_parallel_efficiency: Optional[float] = None  # 0-1
    scalability_tested: bool = False
    max_tested_cores: int = 1


@dataclass
class PerformanceProfile:
    """Podaci o performansama (ako postoje)."""
    single_core_time_seconds: float
    parallel_time_seconds: float
    cores_used: int
    memory_peak_gb: float
    io_time_percentage: float
    communication_time_percentage: float = 0


class SoftwareAnalyzer:
    """
    Analizator softverskih rješenja za HPC spremnost.
    Procjenjuje skalabilnost, paralelizaciju i uska grla.
    """
    
    def __init__(self, profile: SoftwareProfile):
        self.profile = profile
        self.analysis_results = {}
        self.code_metrics = CodeMetrics()
    
    def analyze(self) -> Dict:
        """Izvrši kompletnu analizu softvera."""
        # Analiza koda ako je dostupan
        if self.profile.source_code:
            self.code_metrics = self._analyze_code()
        
        self.analysis_results = {
            "code_metrics": self._get_code_metrics_dict(),
            "parallelization_assessment": self._assess_parallelization(),
            "scalability_assessment": self._assess_scalability(),
            "optimization_opportunities": self._find_optimization_opportunities(),
            "antipatterns_detected": self._detect_antipatterns(),
            "positive_patterns": self._detect_positive_patterns(),
            "recommendations": self._generate_recommendations(),
            "software_score": 0
        }
        
        self.analysis_results["software_score"] = self._calculate_software_score()
        
        return self.analysis_results
    
    def _analyze_code(self) -> CodeMetrics:
        """Analiziraj izvorni kod."""
        code = self.profile.source_code
        metrics = CodeMetrics()
        
        lines = code.split('\n')
        metrics.total_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # Detekcija petlji
        loop_patterns = [
            r'\bfor\s+', r'\bwhile\s+', r'\bdo\s*{',
            r'\.forEach\(', r'\.map\(', r'\.reduce\('
        ]
        for pattern in loop_patterns:
            metrics.loop_count += len(re.findall(pattern, code))
        
        # Detekcija paralelnih petlji
        parallel_patterns = [
            r'#pragma\s+omp\s+parallel',
            r'@parallel', r'@njit.*parallel',
            r'Pool\(', r'ThreadPoolExecutor',
            r'multiprocessing\.',
            r'Parallel\('
        ]
        for pattern in parallel_patterns:
            metrics.parallel_loop_count += len(re.findall(pattern, code))
        
        # MPI pozivi
        mpi_patterns = [
            r'MPI_', r'mpi4py', r'from\s+mpi4py',
            r'MPI\.COMM_WORLD', r'comm\.(send|recv|bcast|scatter|gather)'
        ]
        for pattern in mpi_patterns:
            metrics.mpi_calls += len(re.findall(pattern, code, re.IGNORECASE))
        
        # OpenMP pragma
        metrics.openmp_pragmas = len(re.findall(r'#pragma\s+omp', code))
        
        # CUDA kerneli
        cuda_patterns = [r'__global__', r'<<<.*>>>', r'cuda\.', r'cupy\.']
        for pattern in cuda_patterns:
            metrics.cuda_kernels += len(re.findall(pattern, code))
        
        # I/O operacije
        io_patterns = [
            r'open\(', r'read\(', r'write\(',
            r'fopen', r'fread', r'fwrite',
            r'print\(', r'input\(',
            r'\.to_csv\(', r'\.read_csv\(',
            r'\.save\(', r'\.load\('
        ]
        for pattern in io_patterns:
            metrics.io_operations += len(re.findall(pattern, code))
        
        # Memory alokacije
        mem_patterns = [
            r'malloc\(', r'calloc\(', r'realloc\(',
            r'new\s+\w+\[', r'np\.zeros\(', r'np\.empty\(',
            r'torch\.zeros\(', r'tf\.zeros\('
        ]
        for pattern in mem_patterns:
            metrics.memory_allocations += len(re.findall(pattern, code))
        
        # Sinhronizacija
        sync_patterns = [
            r'MPI_Barrier', r'#pragma\s+omp\s+barrier',
            r'\.join\(\)', r'\.wait\(',
            r'Lock\(\)', r'Semaphore\(',
            r'cuda\.synchronize', r'cudaDeviceSynchronize'
        ]
        for pattern in sync_patterns:
            metrics.synchronization_points += len(re.findall(pattern, code))
        
        # Funkcije
        func_patterns = [r'def\s+\w+\(', r'function\s+\w+', r'\w+\s*\([^)]*\)\s*{']
        for pattern in func_patterns:
            metrics.function_count += len(re.findall(pattern, code))
        
        return metrics
    
    def _get_code_metrics_dict(self) -> Dict:
        """Konvertuj metrike u dictionary."""
        m = self.code_metrics
        return {
            "total_lines": m.total_lines,
            "loop_count": m.loop_count,
            "parallel_loop_count": m.parallel_loop_count,
            "function_count": m.function_count,
            "mpi_calls": m.mpi_calls,
            "openmp_pragmas": m.openmp_pragmas,
            "cuda_kernels": m.cuda_kernels,
            "io_operations": m.io_operations,
            "memory_allocations": m.memory_allocations,
            "synchronization_points": m.synchronization_points,
            "parallelization_ratio": (
                m.parallel_loop_count / m.loop_count if m.loop_count > 0 else 0
            )
        }
    
    def _assess_parallelization(self) -> Dict:
        """Procjena nivoa paralelizacije."""
        score = 0
        level = ParallelizationLevel.NONE
        details = []
        
        # Na osnovu deklarisanih tehnologija
        if self.profile.uses_mpi:
            score += 30
            details.append("Koristi MPI za distribuiranu paralelizaciju")
        
        if self.profile.uses_openmp:
            score += 20
            details.append("Koristi OpenMP za shared-memory paralelizaciju")
        
        if self.profile.uses_cuda:
            score += 25
            details.append("Koristi CUDA za GPU akceleraciju")
        
        if self.profile.uses_vectorization:
            score += 15
            details.append("Koristi vektorizaciju (SIMD)")
        
        # Na osnovu analize koda
        m = self.code_metrics
        if m.parallel_loop_count > 0:
            parallelization_ratio = m.parallel_loop_count / max(m.loop_count, 1)
            score += parallelization_ratio * 10
            details.append(f"Paralelizovano {parallelization_ratio*100:.1f}% petlji")
        
        # Odredi nivo
        if score >= 80:
            level = ParallelizationLevel.OPTIMAL
        elif score >= 60:
            level = ParallelizationLevel.ADVANCED
        elif score >= 40:
            level = ParallelizationLevel.MODERATE
        elif score >= 20:
            level = ParallelizationLevel.BASIC
        else:
            level = ParallelizationLevel.NONE
        
        return {
            "level": level.name,
            "score": min(100, score),
            "uses_mpi": self.profile.uses_mpi,
            "uses_openmp": self.profile.uses_openmp,
            "uses_cuda": self.profile.uses_cuda,
            "uses_vectorization": self.profile.uses_vectorization,
            "details": details
        }
    
    def _assess_scalability(self) -> Dict:
        """Procjena skalabilnosti."""
        assessment = {
            "tested": self.profile.scalability_tested,
            "max_tested_cores": self.profile.max_tested_cores,
            "estimated_scalability": "unknown",
            "parallel_efficiency": None,
            "amdahl_limit": None,
            "gustafson_potential": None
        }
        
        if self.profile.measured_parallel_efficiency is not None:
            eff = self.profile.measured_parallel_efficiency
            assessment["parallel_efficiency"] = eff
            
            if eff >= 0.8:
                assessment["estimated_scalability"] = "excellent"
            elif eff >= 0.6:
                assessment["estimated_scalability"] = "good"
            elif eff >= 0.4:
                assessment["estimated_scalability"] = "moderate"
            else:
                assessment["estimated_scalability"] = "poor"
            
            # Procjena Amdahl-ovog limita
            # Ako je efikasnost e na P procesora, sekvencijalni dio je približno: s = (1/e - 1)/(P-1) * P
            if self.profile.max_tested_cores > 1 and eff > 0:
                P = self.profile.max_tested_cores
                # Pojednostavljena procjena sekvencijalnog dijela
                seq_fraction = max(0, 1 - eff) / (1 - 1/P) if P > 1 else 1
                seq_fraction = min(1, seq_fraction)
                
                # Amdahl-ov limit (za beskonačno procesora)
                if seq_fraction > 0:
                    amdahl_limit = 1 / seq_fraction
                else:
                    amdahl_limit = float('inf')
                
                assessment["amdahl_limit"] = min(1000, amdahl_limit)
                assessment["sequential_fraction"] = seq_fraction
        else:
            # Procjena na osnovu karakteristika koda
            m = self.code_metrics
            if m.loop_count > 0:
                par_ratio = m.parallel_loop_count / m.loop_count
                sync_ratio = m.synchronization_points / max(m.parallel_loop_count, 1)
                io_ratio = m.io_operations / max(m.total_lines, 1)
                
                # Jednostavna heuristika za procjenu skalabilnosti
                scalability_estimate = par_ratio * 0.5 - sync_ratio * 0.3 - io_ratio * 0.2
                
                if scalability_estimate > 0.3:
                    assessment["estimated_scalability"] = "potentially_good"
                elif scalability_estimate > 0:
                    assessment["estimated_scalability"] = "potentially_moderate"
                else:
                    assessment["estimated_scalability"] = "potentially_poor"
        
        return assessment
    
    def _detect_antipatterns(self) -> List[Dict]:
        """Detekcija anti-pattern-a u kodu."""
        detected = []
        code = self.profile.source_code or ""
        m = self.code_metrics
        
        # Sequential loops bez paralelizacije
        if m.loop_count > 0 and m.parallel_loop_count == 0:
            detected.append({
                "pattern": "sequential_loops",
                **SOFTWARE_ANTIPATTERNS["sequential_loops"]
            })
        
        # Global variables (Python specific)
        global_count = len(re.findall(r'\bglobal\s+\w+', code))
        if global_count > 3:
            detected.append({
                "pattern": "global_variables",
                **SOFTWARE_ANTIPATTERNS["global_variables"],
                "count": global_count
            })
        
        # Blocking I/O u petljama
        io_in_loop = bool(re.search(
            r'(for|while)[^{]*{[^}]*(open\(|read\(|write\()', 
            code, re.DOTALL
        ))
        if io_in_loop:
            detected.append({
                "pattern": "blocking_io",
                **SOFTWARE_ANTIPATTERNS["blocking_io"]
            })
        
        # Excessive synchronization
        if m.synchronization_points > m.parallel_loop_count * 2 and m.parallel_loop_count > 0:
            detected.append({
                "pattern": "excessive_synchronization",
                **SOFTWARE_ANTIPATTERNS["excessive_synchronization"]
            })
        
        # Nested loops (potencijalno O(n²))
        nested_loops = len(re.findall(
            r'for[^{]*{[^}]*for[^{]*{', 
            code, re.DOTALL
        ))
        if nested_loops > 0:
            detected.append({
                "pattern": "unoptimized_algorithms",
                **SOFTWARE_ANTIPATTERNS["unoptimized_algorithms"],
                "nested_loop_count": nested_loops
            })
        
        return detected
    
    def _detect_positive_patterns(self) -> List[Dict]:
        """Detekcija pozitivnih pattern-a."""
        detected = []
        code = self.profile.source_code or ""
        
        # Vectorized operations
        if re.search(r'(np\.|numpy\.|@vectorize|SIMD)', code):
            detected.append({
                "pattern": "vectorized_operations",
                **SOFTWARE_PATTERNS["vectorized_operations"]
            })
        
        # Parallel loops
        if re.search(r'(#pragma\s+omp|@parallel|Parallel\()', code):
            detected.append({
                "pattern": "parallel_loops",
                **SOFTWARE_PATTERNS["parallel_loops"]
            })
        
        # Async I/O
        if re.search(r'(async\s+def|asyncio|aiofiles|await\s+)', code):
            detected.append({
                "pattern": "async_io",
                **SOFTWARE_PATTERNS["async_io"]
            })
        
        # GPU offloading
        if re.search(r'(__global__|cuda\.|cupy\.|@cuda\.jit)', code):
            detected.append({
                "pattern": "gpu_offloading",
                **SOFTWARE_PATTERNS["gpu_offloading"]
            })
        
        # Memory pooling
        if re.search(r'(memoryview|buffer|pool|preallocate)', code, re.IGNORECASE):
            detected.append({
                "pattern": "memory_pooling",
                **SOFTWARE_PATTERNS["memory_pooling"]
            })
        
        return detected
    
    def _find_optimization_opportunities(self) -> List[Dict]:
        """Pronađi mogućnosti za optimizaciju."""
        opportunities = []
        m = self.code_metrics
        
        # Paralelizacija petlji
        if m.loop_count > m.parallel_loop_count:
            unparallelized = m.loop_count - m.parallel_loop_count
            opportunities.append({
                "type": "loop_parallelization",
                "priority": "high",
                "description": f"{unparallelized} petlji kandidati za paralelizaciju",
                "potential_speedup": f"Do {min(8, unparallelized)}x na multi-core",
                "effort": "medium",
                "recommendation": "Koristiti OpenMP #pragma omp parallel for ili Python multiprocessing"
            })
        
        # GPU offloading
        if not self.profile.uses_cuda and m.loop_count > 5:
            opportunities.append({
                "type": "gpu_offloading",
                "priority": "medium",
                "description": "Potencijal za GPU akceleraciju",
                "potential_speedup": "10-100x za paralelizabilne operacije",
                "effort": "high",
                "recommendation": "Razmotriti CUDA/CuPy za compute-intensive operacije"
            })
        
        # Vektorizacija
        if not self.profile.uses_vectorization:
            opportunities.append({
                "type": "vectorization",
                "priority": "medium",
                "description": "Potencijal za SIMD vektorizaciju",
                "potential_speedup": "2-8x za numeričke operacije",
                "effort": "low",
                "recommendation": "Koristiti NumPy vektorizovane operacije umjesto Python petlji"
            })
        
        # I/O optimizacija
        if m.io_operations > m.total_lines * 0.05:  # >5% I/O
            opportunities.append({
                "type": "io_optimization",
                "priority": "high",
                "description": "Visok udio I/O operacija",
                "potential_speedup": "Smanjenje I/O bottleneck-a",
                "effort": "medium",
                "recommendation": "Koristiti async I/O, buffering, ili parallel I/O (MPI-IO)"
            })
        
        # MPI za distribuiranu paralelizaciju
        if not self.profile.uses_mpi and m.total_lines > 500:
            opportunities.append({
                "type": "distributed_computing",
                "priority": "medium",
                "description": "Mogućnost distribucije preko čvorova",
                "potential_speedup": "Skaliranje preko jednog čvora",
                "effort": "high",
                "recommendation": "Implementirati MPI za large-scale probleme"
            })
        
        return opportunities
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generiši konkretne preporuke za optimizaciju."""
        recommendations = []
        
        antipatterns = self._detect_antipatterns()
        opportunities = self._find_optimization_opportunities()
        
        # Preporuke bazirane na anti-pattern-ima
        for ap in antipatterns:
            recommendations.append({
                "priority": 1 if ap["penalty"] > 15 else 2,
                "type": "fix_antipattern",
                "issue": ap["description"],
                "action": f"Refaktorisati kod za uklanjanje: {ap['pattern']}",
                "expected_improvement": f"Smanjenje penala za {ap['penalty']} poena"
            })
        
        # Preporuke bazirane na mogućnostima
        for opp in opportunities:
            if opp["priority"] == "high":
                recommendations.append({
                    "priority": 1,
                    "type": "optimization",
                    "issue": opp["description"],
                    "action": opp["recommendation"],
                    "expected_improvement": opp["potential_speedup"]
                })
        
        # Specifične preporuke za HPC
        if not self.profile.has_profiling_data:
            recommendations.append({
                "priority": 1,
                "type": "profiling",
                "issue": "Nedostaju podaci o profilisanju",
                "action": "Profilisati aplikaciju sa Intel VTune, gprof, ili Nsight",
                "expected_improvement": "Identifikacija hotspots i bottlenecks"
            })
        
        if not self.profile.scalability_tested:
            recommendations.append({
                "priority": 1,
                "type": "testing",
                "issue": "Nije testirana skalabilnost",
                "action": "Izvršiti strong i weak scaling testove",
                "expected_improvement": "Razumijevanje scaling limita"
            })
        
        recommendations.sort(key=lambda x: x["priority"])
        
        return recommendations
    
    def _calculate_software_score(self) -> float:
        """Izračunaj ukupni software optimization score."""
        base_score = 50  # Početni score
        
        # Parallelization bonus
        par_score = self.analysis_results["parallelization_assessment"]["score"]
        base_score += par_score * 0.3
        
        # Scalability bonus
        scalability = self.analysis_results["scalability_assessment"]
        if scalability["estimated_scalability"] == "excellent":
            base_score += 20
        elif scalability["estimated_scalability"] == "good":
            base_score += 15
        elif scalability["estimated_scalability"] in ["moderate", "potentially_good"]:
            base_score += 10
        elif scalability["estimated_scalability"] == "potentially_moderate":
            base_score += 5
        
        # Positive patterns bonus
        for pattern in self.analysis_results["positive_patterns"]:
            base_score += pattern["bonus"] * 0.5
        
        # Antipattern penalties
        for ap in self.analysis_results["antipatterns_detected"]:
            base_score -= ap["penalty"] * 0.5
        
        return max(0, min(100, base_score))
    
    def get_summary(self) -> str:
        """Generiši tekstualni sažetak analize."""
        if not self.analysis_results:
            self.analyze()
        
        r = self.analysis_results
        par = r["parallelization_assessment"]
        
        summary = f"""
═══════════════════════════════════════════════════════════════
            ANALIZA SOFTVERA: {self.profile.name}
═══════════════════════════════════════════════════════════════

OSNOVNE INFORMACIJE
   - Jezik: {self.profile.language}
   - Linije koda: {r['code_metrics'].get('total_lines', self.profile.estimated_loc)}
   - Broj funkcija: {r['code_metrics'].get('function_count', 'N/A')}

PARALELIZACIJA (Score: {par['score']:.1f}/100)
   - Nivo: {par['level']}
   - MPI: {'Da' if par['uses_mpi'] else 'Ne'}
   - OpenMP: {'Da' if par['uses_openmp'] else 'Ne'}
   - CUDA: {'Da' if par['uses_cuda'] else 'Ne'}
   - Vektorizacija: {'Da' if par['uses_vectorization'] else 'Ne'}
   - Petlje: {r['code_metrics'].get('loop_count', 0)} ukupno, {r['code_metrics'].get('parallel_loop_count', 0)} paralelno

SKALABILNOST
   - Procjena: {r['scalability_assessment']['estimated_scalability']}
   - Testirano: {'Da' if r['scalability_assessment']['tested'] else 'Ne'}
   - Max testiranih jezgara: {r['scalability_assessment']['max_tested_cores']}
"""
        
        if r['scalability_assessment'].get('parallel_efficiency'):
            summary += f"   - Paralelna efikasnost: {r['scalability_assessment']['parallel_efficiency']*100:.1f}%\n"
        
        summary += f"""
 ANTI-PATTERNS DETEKTOVANI: {len(r['antipatterns_detected'])}
"""
        for ap in r['antipatterns_detected'][:3]:
            summary += f"   - {ap['description']} (penalizacija: -{ap['penalty']})\n"
        
        summary += f"""
POZITIVNI PATTERNS: {len(r['positive_patterns'])}
"""
        for pp in r['positive_patterns'][:3]:
            summary += f"   - {pp['description']} (bonus: +{pp['bonus']})\n"
        
        summary += f"""
TOP MOGUĆNOSTI ZA OPTIMIZACIJU
"""
        for opp in r['optimization_opportunities'][:3]:
            summary += f"   - [{opp['priority'].upper()}] {opp['description']}\n"
            summary += f"     → {opp['recommendation']}\n"
        
        summary += f"""
═══════════════════════════════════════════════════════════════
            SOFTWARE OPTIMIZATION SCORE: {r['software_score']:.1f}/100
═══════════════════════════════════════════════════════════════
"""
        return summary
