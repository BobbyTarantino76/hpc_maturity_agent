"""
Modul za analizu infrastrukture i radnih opterećenja
=====================================================
Prikupljanje i obrada podataka o postojećoj infrastrukturi,
metrike iskorištenosti resursa i karakterizacija workload-a.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import statistics


class ResourceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ComputeNode:
    """Reprezentacija računarskog čvora."""
    node_id: str
    cpu_cores: int
    cpu_frequency_ghz: float
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0
    storage_tb: float = 0
    network_bandwidth_gbps: float = 1.0
    
    def total_compute_capacity(self) -> float:
        """Procjena ukupnog računarskog kapaciteta (relativne jedinice)."""
        cpu_capacity = self.cpu_cores * self.cpu_frequency_ghz
        gpu_capacity = self.gpu_count * 50  # GPU ekvivalent ~50 CPU jezgara
        return cpu_capacity + gpu_capacity


@dataclass
class Workload:
    """Reprezentacija radnog opterećenja."""
    workload_id: str
    name: str
    workload_type: str  # compute_intensive, memory_intensive, io_intensive, balanced, gpu_accelerated
    avg_cpu_utilization: float  # 0-100%
    avg_memory_utilization: float  # 0-100%
    avg_io_operations_per_sec: float
    avg_runtime_hours: float
    parallelizable_fraction: float  # Amdahl's law faktor (0-1)
    data_size_gb: float
    frequency_per_month: int  # Koliko puta mjesečno se izvršava
    
    def compute_intensity_score(self) -> float:
        """Izračunaj intenzitet računanja."""
        return (self.avg_cpu_utilization * 0.6 + 
                (1 - self.avg_memory_utilization/100) * 0.2 +
                self.parallelizable_fraction * 100 * 0.2)


@dataclass
class ResourceMetrics:
    """Metrike iskorištenosti resursa."""
    avg_cpu_utilization: float
    peak_cpu_utilization: float
    avg_memory_utilization: float
    peak_memory_utilization: float
    avg_gpu_utilization: float = 0
    avg_storage_utilization: float = 0
    avg_network_utilization: float = 0
    idle_time_percentage: float = 0


@dataclass 
class InfrastructureProfile:
    """Kompletan profil infrastrukture."""
    organization_name: str
    nodes: List[ComputeNode] = field(default_factory=list)
    workloads: List[Workload] = field(default_factory=list)
    metrics: Optional[ResourceMetrics] = None
    has_job_scheduler: bool = False
    scheduler_type: str = "none"  # slurm, pbs, sge, none
    has_shared_filesystem: bool = False
    filesystem_type: str = "local"  # lustre, gpfs, nfs, local
    interconnect_type: str = "ethernet"  # infiniband, ethernet, omnipath
    
    def total_cpu_cores(self) -> int:
        return sum(node.cpu_cores for node in self.nodes)
    
    def total_memory_gb(self) -> float:
        return sum(node.memory_gb for node in self.nodes)
    
    def total_gpus(self) -> int:
        return sum(node.gpu_count for node in self.nodes)


class InfrastructureAnalyzer:
    """
    Analizator infrastrukture i workload-a.
    Prikuplja podatke, izračunava metrike i ocjenjuje spremnost.
    """
    
    def __init__(self, profile: InfrastructureProfile):
        self.profile = profile
        self.analysis_results = {}
    
    def analyze(self) -> Dict:
        """Izvrši kompletnu analizu infrastrukture."""
        self.analysis_results = {
            "capacity_analysis": self._analyze_capacity(),
            "utilization_analysis": self._analyze_utilization(),
            "workload_analysis": self._analyze_workloads(),
            "hpc_readiness": self._assess_hpc_readiness(),
            "bottlenecks": self._identify_bottlenecks(),
            "infrastructure_score": 0
        }
        
        # Izračunaj ukupni score
        self.analysis_results["infrastructure_score"] = self._calculate_infrastructure_score()
        
        return self.analysis_results
    
    def _analyze_capacity(self) -> Dict:
        """Analiza ukupnog kapaciteta."""
        if not self.profile.nodes:
            return {"status": "no_nodes", "total_capacity": 0}
        
        total_compute = sum(n.total_compute_capacity() for n in self.profile.nodes)
        
        return {
            "total_nodes": len(self.profile.nodes),
            "total_cpu_cores": self.profile.total_cpu_cores(),
            "total_memory_gb": self.profile.total_memory_gb(),
            "total_gpus": self.profile.total_gpus(),
            "total_compute_capacity": total_compute,
            "avg_cores_per_node": statistics.mean([n.cpu_cores for n in self.profile.nodes]),
            "avg_memory_per_node": statistics.mean([n.memory_gb for n in self.profile.nodes]),
            "heterogeneous": self._is_heterogeneous()
        }
    
    def _is_heterogeneous(self) -> bool:
        """Provjeri da li je klaster heterogen."""
        if len(self.profile.nodes) < 2:
            return False
        cores = [n.cpu_cores for n in self.profile.nodes]
        return max(cores) != min(cores)
    
    def _analyze_utilization(self) -> Dict:
        """Analiza iskorištenosti resursa."""
        if not self.profile.metrics:
            return {"status": "no_metrics", "efficiency": 0}
        
        m = self.profile.metrics
        
        # Efikasnost = prosječna iskorištenost bez preopterećenja
        efficiency = min(m.avg_cpu_utilization, 85) / 85 * 100
        
        return {
            "cpu_efficiency": efficiency,
            "memory_pressure": m.peak_memory_utilization > 90,
            "idle_waste": m.idle_time_percentage,
            "gpu_utilization": m.avg_gpu_utilization,
            "balanced_utilization": abs(m.avg_cpu_utilization - m.avg_memory_utilization) < 20,
            "overprovisioned": m.avg_cpu_utilization < 30 and m.avg_memory_utilization < 30
        }
    
    def _analyze_workloads(self) -> Dict:
        """Analiza karakteristika workload-a."""
        if not self.profile.workloads:
            return {"status": "no_workloads", "hpc_suitable": 0}
        
        workloads = self.profile.workloads
        
        # Klasifikacija workload-a
        type_distribution = {}
        for w in workloads:
            type_distribution[w.workload_type] = type_distribution.get(w.workload_type, 0) + 1
        
        # Prosječna paralelizabilnost
        avg_parallelizable = statistics.mean([w.parallelizable_fraction for w in workloads])
        
        # Računanje potencijalnog speedup-a (Amdahl's law za 64 procesora)
        potential_speedups = []
        for w in workloads:
            p = w.parallelizable_fraction
            speedup = 1 / ((1 - p) + p / 64)
            potential_speedups.append(speedup)
        
        avg_speedup = statistics.mean(potential_speedups)
        
        # HPC pogodnost - visoko paralelizabilni workload-i
        hpc_suitable = sum(1 for w in workloads if w.parallelizable_fraction > 0.7)
        
        return {
            "total_workloads": len(workloads),
            "type_distribution": type_distribution,
            "avg_parallelizable_fraction": avg_parallelizable,
            "avg_potential_speedup_64p": avg_speedup,
            "hpc_suitable_count": hpc_suitable,
            "hpc_suitable_percentage": hpc_suitable / len(workloads) * 100,
            "total_monthly_compute_hours": sum(w.avg_runtime_hours * w.frequency_per_month for w in workloads),
            "largest_workload_data_gb": max(w.data_size_gb for w in workloads)
        }
    
    def _assess_hpc_readiness(self) -> Dict:
        """Procjena spremnosti za HPC."""
        scores = {}
        
        # Job scheduler (0-20 poena)
        if self.profile.has_job_scheduler:
            scheduler_scores = {"slurm": 20, "pbs": 18, "sge": 15}
            scores["scheduler"] = scheduler_scores.get(self.profile.scheduler_type, 10)
        else:
            scores["scheduler"] = 0
        
        # Shared filesystem (0-20 poena)
        if self.profile.has_shared_filesystem:
            fs_scores = {"lustre": 20, "gpfs": 20, "nfs": 10}
            scores["filesystem"] = fs_scores.get(self.profile.filesystem_type, 5)
        else:
            scores["filesystem"] = 0
        
        # Interconnect (0-20 poena)
        interconnect_scores = {"infiniband": 20, "omnipath": 18, "ethernet": 8}
        scores["interconnect"] = interconnect_scores.get(self.profile.interconnect_type, 5)
        
        # Skala klastera (0-20 poena)
        node_count = len(self.profile.nodes)
        if node_count >= 100:
            scores["scale"] = 20
        elif node_count >= 50:
            scores["scale"] = 15
        elif node_count >= 10:
            scores["scale"] = 10
        elif node_count >= 4:
            scores["scale"] = 5
        else:
            scores["scale"] = 2
        
        # GPU kapacitet (0-20 poena)
        gpu_count = self.profile.total_gpus()
        if gpu_count >= 32:
            scores["gpu"] = 20
        elif gpu_count >= 8:
            scores["gpu"] = 15
        elif gpu_count >= 2:
            scores["gpu"] = 10
        elif gpu_count >= 1:
            scores["gpu"] = 5
        else:
            scores["gpu"] = 0
        
        total = sum(scores.values())
        
        return {
            "component_scores": scores,
            "total_score": total,
            "max_score": 100,
            "readiness_percentage": total
        }
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """Identifikacija uskih grla."""
        bottlenecks = []
        
        # Memory bottleneck
        if self.profile.metrics:
            if self.profile.metrics.peak_memory_utilization > 90:
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high",
                    "description": "Vršna memorija prelazi 90%, rizik od swapping-a",
                    "recommendation": "Povećati RAM ili optimizovati memorijski footprint"
                })
        
        # I/O bottleneck
        if self.profile.filesystem_type == "local":
            bottlenecks.append({
                "type": "storage",
                "severity": "medium",
                "description": "Lokalni storage ograničava skalabilnost",
                "recommendation": "Implementirati dijeljeni filesystem (Lustre, GPFS)"
            })
        
        # Network bottleneck
        if self.profile.interconnect_type == "ethernet":
            bottlenecks.append({
                "type": "network",
                "severity": "medium", 
                "description": "Ethernet interconnect ima visoku latenciju za MPI",
                "recommendation": "Razmotriti InfiniBand za MPI workload-e"
            })
        
        # Scheduling bottleneck
        if not self.profile.has_job_scheduler:
            bottlenecks.append({
                "type": "scheduling",
                "severity": "high",
                "description": "Nedostaje job scheduler za efikasno upravljanje resursima",
                "recommendation": "Implementirati SLURM ili PBS za job scheduling"
            })
        
        return bottlenecks
    
    def _calculate_infrastructure_score(self) -> float:
        """Izračunaj ukupni infrastructure score (0-100)."""
        scores = []
        
        # HPC readiness score (40% težine)
        hpc_score = self.analysis_results["hpc_readiness"]["readiness_percentage"]
        scores.append(hpc_score * 0.4)
        
        # Workload suitability (30% težine)
        workload_analysis = self.analysis_results["workload_analysis"]
        if workload_analysis.get("status") != "no_workloads":
            workload_score = workload_analysis["hpc_suitable_percentage"]
            scores.append(workload_score * 0.3)
        else:
            scores.append(0)
        
        # Utilization efficiency (20% težine)
        util_analysis = self.analysis_results["utilization_analysis"]
        if util_analysis.get("status") != "no_metrics":
            util_score = util_analysis["cpu_efficiency"]
            scores.append(util_score * 0.2)
        else:
            scores.append(10)  # Defaultna vrijednost ako nema metrika
        
        # Bottleneck penalty (10% težine)
        bottleneck_count = len(self.analysis_results["bottlenecks"])
        bottleneck_score = max(0, 100 - bottleneck_count * 25)
        scores.append(bottleneck_score * 0.1)
        
        return sum(scores)
    
    def get_summary(self) -> str:
        """Generiši tekstualni sažetak analize."""
        if not self.analysis_results:
            self.analyze()
        
        r = self.analysis_results
        
        summary = f"""
═══════════════════════════════════════════════════════════════
          ANALIZA INFRASTRUKTURE: {self.profile.organization_name}
═══════════════════════════════════════════════════════════════

KAPACITET
   - Ukupno čvorova: {r['capacity_analysis'].get('total_nodes', 0)}
   - CPU jezgara: {r['capacity_analysis'].get('total_cpu_cores', 0)}
   - Memorija: {r['capacity_analysis'].get('total_memory_gb', 0):.1f} GB
   - GPU: {r['capacity_analysis'].get('total_gpus', 0)}

ISKORIŠTENOST
   - CPU efikasnost: {r['utilization_analysis'].get('cpu_efficiency', 0):.1f}%
   - Memory pressure: {'Da' if r['utilization_analysis'].get('memory_pressure') else 'Ne'}
   - Balansirano: {'Da' if r['utilization_analysis'].get('balanced_utilization') else 'Ne'}

WORKLOAD KARAKTERISTIKE
   - Broj workload-a: {r['workload_analysis'].get('total_workloads', 0)}
   - HPC pogodni: {r['workload_analysis'].get('hpc_suitable_percentage', 0):.1f}%
   - Prosječni speedup (64P): {r['workload_analysis'].get('avg_potential_speedup_64p', 1):.2f}x

HPC SPREMNOST
   - Score: {r['hpc_readiness']['total_score']}/100
   - Scheduler: {self.profile.scheduler_type}
   - Filesystem: {self.profile.filesystem_type}
   - Interconnect: {self.profile.interconnect_type}

 USKA GRLA: {len(r['bottlenecks'])} identifikovano

═══════════════════════════════════════════════════════════════
   INFRASTRUCTURE SCORE: {r['infrastructure_score']:.1f}/100
═══════════════════════════════════════════════════════════════
"""
        return summary
