"""
Modul za simulaciju kapaciteta i resursa
=========================================
Model za procjenu performansi, zauzeće resursa i troškova
za modelirane workload-e na HPC infrastrukturi.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random

from config import SIMULATION_PARAMS, WORKLOAD_TYPES


@dataclass
class SimulationWorkload:
    """Workload za simulaciju."""
    name: str
    workload_type: str
    base_runtime_hours: float
    cpu_cores_required: int
    memory_gb_required: float
    gpu_required: int = 0
    data_size_gb: float = 0
    parallelizable_fraction: float = 0.8
    io_intensity: float = 0.1  # 0-1
    monthly_runs: int = 1


@dataclass
class HPCClusterConfig:
    """Konfiguracija HPC klastera za simulaciju."""
    name: str
    total_nodes: int
    cores_per_node: int
    memory_per_node_gb: float
    gpus_per_node: int = 0
    interconnect_bandwidth_gbps: float = 100
    storage_capacity_tb: float = 100
    parallel_efficiency: float = 0.85


@dataclass
class SimulationResult:
    """Rezultat simulacije."""
    workload_name: str
    estimated_runtime_hours: float
    speedup: float
    parallel_efficiency: float
    resource_utilization: Dict[str, float]
    estimated_cost: float
    bottleneck: Optional[str]
    recommendations: List[str]


class HPCSimulator:
    """
    Simulator HPC performansi i resursa.
    Koristi Amdahl-ov i Gustafson-ov zakon za procjenu skaliranja.
    """
    
    def __init__(self, cluster: HPCClusterConfig):
        self.cluster = cluster
        self.simulation_results = []
    
    def simulate_workload(self, workload: SimulationWorkload, 
                         allocated_cores: Optional[int] = None) -> SimulationResult:
        """Simuliraj izvršenje workload-a na klasteru."""
        
        # Odredi broj alocirani jezgara
        if allocated_cores is None:
            allocated_cores = min(
                workload.cpu_cores_required,
                self.cluster.total_nodes * self.cluster.cores_per_node
            )
        
        # Izračunaj speedup koristeći Amdahl-ov zakon
        p = workload.parallelizable_fraction
        n = allocated_cores
        
        # Amdahl's law: S = 1 / ((1-p) + p/n)
        theoretical_speedup = 1 / ((1 - p) + p / n)
        
        # Primijeni overhead i efikasnost klastera
        overhead = SIMULATION_PARAMS["parallel_overhead"]
        cluster_efficiency = self.cluster.parallel_efficiency
        
        actual_speedup = theoretical_speedup * cluster_efficiency * (1 - overhead)
        
        # Procijeni runtime
        estimated_runtime = workload.base_runtime_hours / actual_speedup
        
        # Paralelna efikasnost
        parallel_efficiency = actual_speedup / n if n > 0 else 0
        
        # Procjena korištenja resursa
        resource_utilization = self._calculate_resource_utilization(
            workload, allocated_cores
        )
        
        # Procjena troškova
        estimated_cost = self._calculate_cost(
            workload, estimated_runtime, allocated_cores, resource_utilization
        )
        
        # Identifikuj bottleneck
        bottleneck = self._identify_bottleneck(workload, resource_utilization)
        
        # Generiši preporuke
        recommendations = self._generate_simulation_recommendations(
            workload, speedup=actual_speedup, efficiency=parallel_efficiency,
            bottleneck=bottleneck
        )
        
        result = SimulationResult(
            workload_name=workload.name,
            estimated_runtime_hours=estimated_runtime,
            speedup=actual_speedup,
            parallel_efficiency=parallel_efficiency,
            resource_utilization=resource_utilization,
            estimated_cost=estimated_cost,
            bottleneck=bottleneck,
            recommendations=recommendations
        )
        
        self.simulation_results.append(result)
        
        return result
    
    def _calculate_resource_utilization(self, workload: SimulationWorkload,
                                        allocated_cores: int) -> Dict[str, float]:
        """Izračunaj procjenu korištenja resursa."""
        total_cores = self.cluster.total_nodes * self.cluster.cores_per_node
        total_memory = self.cluster.total_nodes * self.cluster.memory_per_node_gb
        total_gpus = self.cluster.total_nodes * self.cluster.gpus_per_node
        
        # CPU utilization
        cpu_util = min(100, (allocated_cores / total_cores) * 100)
        
        # Memory utilization
        nodes_used = math.ceil(allocated_cores / self.cluster.cores_per_node)
        memory_available = nodes_used * self.cluster.memory_per_node_gb
        memory_util = min(100, (workload.memory_gb_required / memory_available) * 100)
        
        # GPU utilization
        gpu_util = 0
        if total_gpus > 0 and workload.gpu_required > 0:
            gpu_util = min(100, (workload.gpu_required / total_gpus) * 100)
        
        # I/O utilization (procjena)
        io_util = workload.io_intensity * 100
        
        # Network utilization (procjena bazirana na paralelizaciji)
        # Više paralelizacije = više komunikacije
        network_util = workload.parallelizable_fraction * allocated_cores / total_cores * 50
        
        return {
            "cpu": cpu_util,
            "memory": memory_util,
            "gpu": gpu_util,
            "io": io_util,
            "network": network_util,
            "nodes_used": nodes_used,
            "cores_used": allocated_cores
        }
    
    def _calculate_cost(self, workload: SimulationWorkload,
                       runtime_hours: float, cores: int,
                       utilization: Dict[str, float]) -> float:
        """Izračunaj procijenjeni trošak izvršenja."""
        params = SIMULATION_PARAMS
        
        # CPU trošak
        cpu_cost = cores * runtime_hours * params["cpu_cost_per_hour"]
        
        # GPU trošak
        gpu_cost = workload.gpu_required * runtime_hours * params["gpu_cost_per_hour"]
        
        # Memory trošak
        memory_cost = (workload.memory_gb_required * runtime_hours * 
                      params["memory_cost_per_gb_hour"])
        
        # Storage trošak (proporcionalno veličini podataka, mjesečno)
        storage_cost = (workload.data_size_gb / 1000) * params["storage_cost_per_tb_month"] / 30
        
        # Network trošak (procjena transfera)
        network_cost = workload.data_size_gb * 0.1 * params["network_cost_per_gb"]
        
        total_cost = cpu_cost + gpu_cost + memory_cost + storage_cost + network_cost
        
        # Multipliciraj sa brojem mjesečnih izvršenja
        monthly_cost = total_cost * workload.monthly_runs
        
        return monthly_cost
    
    def _identify_bottleneck(self, workload: SimulationWorkload,
                            utilization: Dict[str, float]) -> Optional[str]:
        """Identifikuj primarni bottleneck."""
        
        # Sortiraj po utilizaciji
        util_items = [
            ("cpu", utilization["cpu"]),
            ("memory", utilization["memory"]),
            ("io", utilization["io"]),
            ("network", utilization["network"])
        ]
        
        if workload.gpu_required > 0:
            util_items.append(("gpu", utilization["gpu"]))
        
        # Pronađi najveći
        max_item = max(util_items, key=lambda x: x[1])
        
        # Ako je korištenje >80%, to je bottleneck
        if max_item[1] > 80:
            return max_item[0]
        
        # Ako je parallelizable_fraction nizak, CPU je bottleneck
        if workload.parallelizable_fraction < 0.5:
            return "serial_code"
        
        return None
    
    def _generate_simulation_recommendations(self, workload: SimulationWorkload,
                                            speedup: float, efficiency: float,
                                            bottleneck: Optional[str]) -> List[str]:
        """Generiši preporuke bazirane na simulaciji."""
        recommendations = []
        
        if efficiency < 0.5:
            recommendations.append(
                "Niska paralelna efikasnost. Razmotriti smanjenje broja jezgara "
                "ili optimizaciju komunikacije."
            )
        
        if speedup < 2 and workload.parallelizable_fraction > 0.5:
            recommendations.append(
                "Speedup je ispod očekivanog. Profilisati aplikaciju za "
                "identifikaciju overhead-a."
            )
        
        if bottleneck == "memory":
            recommendations.append(
                "Memory bottleneck. Razmotriti out-of-core algoritme ili "
                "povećanje memorije po čvoru."
            )
        elif bottleneck == "io":
            recommendations.append(
                "I/O bottleneck. Koristiti parallel I/O (MPI-IO, HDF5) "
                "ili SSD storage."
            )
        elif bottleneck == "network":
            recommendations.append(
                "Network bottleneck. Optimizovati MPI komunikaciju, "
                "razmotriti InfiniBand interconnect."
            )
        elif bottleneck == "serial_code":
            recommendations.append(
                "Sekvencijalni kod limitira skaliranje. Fokusirati se na "
                "paralelizaciju kritičnih sekcija."
            )
        
        if workload.gpu_required == 0 and workload.workload_type == "compute_intensive":
            recommendations.append(
                "Compute-intensive workload bez GPU-a. Razmotriti GPU akceleraciju "
                "za potencijalno 10-100x ubrzanje."
            )
        
        return recommendations
    
    def run_scaling_study(self, workload: SimulationWorkload,
                         core_counts: List[int] = None) -> Dict:
        """Izvrši scaling studiju za workload."""
        if core_counts is None:
            max_cores = self.cluster.total_nodes * self.cluster.cores_per_node
            core_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            core_counts = [c for c in core_counts if c <= max_cores]
        
        results = []
        for cores in core_counts:
            result = self.simulate_workload(workload, allocated_cores=cores)
            results.append({
                "cores": cores,
                "runtime_hours": result.estimated_runtime_hours,
                "speedup": result.speedup,
                "efficiency": result.parallel_efficiency,
                "cost": result.estimated_cost
            })
        
        # Pronađi optimalni broj jezgara (najbolji cost/performance)
        optimal = min(results, key=lambda x: x["runtime_hours"] * x["cost"])
        
        return {
            "workload": workload.name,
            "scaling_results": results,
            "optimal_cores": optimal["cores"],
            "optimal_runtime": optimal["runtime_hours"],
            "optimal_cost": optimal["cost"],
            "max_speedup": max(r["speedup"] for r in results),
            "efficiency_at_max": results[-1]["efficiency"] if results else 0
        }
    
    def simulate_monthly_usage(self, workloads: List[SimulationWorkload]) -> Dict:
        """Simuliraj mjesečno korištenje klastera."""
        total_compute_hours = 0
        total_cost = 0
        workload_details = []
        
        for wl in workloads:
            result = self.simulate_workload(wl)
            monthly_hours = result.estimated_runtime_hours * wl.monthly_runs
            total_compute_hours += monthly_hours
            total_cost += result.estimated_cost
            
            workload_details.append({
                "name": wl.name,
                "monthly_runs": wl.monthly_runs,
                "hours_per_run": result.estimated_runtime_hours,
                "total_monthly_hours": monthly_hours,
                "monthly_cost": result.estimated_cost,
                "efficiency": result.parallel_efficiency
            })
        
        # Kapacitet klastera (u core-satima mjesečno)
        total_capacity = (self.cluster.total_nodes * 
                         self.cluster.cores_per_node * 
                         24 * 30)  # sati u mjesecu
        
        # Prosječna utilizacija
        avg_utilization = (total_compute_hours / total_capacity) * 100 if total_capacity > 0 else 0
        
        return {
            "cluster": self.cluster.name,
            "total_monthly_compute_hours": total_compute_hours,
            "total_monthly_cost": total_cost,
            "cluster_capacity_hours": total_capacity,
            "average_utilization_percent": avg_utilization,
            "workload_breakdown": workload_details,
            "cost_per_compute_hour": total_cost / total_compute_hours if total_compute_hours > 0 else 0
        }
    
    def get_summary(self, workloads: List[SimulationWorkload]) -> str:
        """Generiši tekstualni sažetak simulacije."""
        monthly = self.simulate_monthly_usage(workloads)
        
        summary = f"""
═══════════════════════════════════════════════════════════════
          SIMULACIJA KAPACITETA: {self.cluster.name}
═══════════════════════════════════════════════════════════════

 KONFIGURACIJA KLASTERA
   - Čvorovi: {self.cluster.total_nodes}
   - Jezgara po čvoru: {self.cluster.cores_per_node}
   - Ukupno jezgara: {self.cluster.total_nodes * self.cluster.cores_per_node}
   - Memorija po čvoru: {self.cluster.memory_per_node_gb} GB
   - GPU po čvoru: {self.cluster.gpus_per_node}
   - Paralelna efikasnost: {self.cluster.parallel_efficiency*100:.0f}%

MJESEČNA PROJEKCIJA
   - Ukupno compute sati: {monthly['total_monthly_compute_hours']:.1f}
   - Kapacitet klastera: {monthly['cluster_capacity_hours']:.0f} core-sati
   - Prosječna utilizacija: {monthly['average_utilization_percent']:.1f}%
   - Ukupni mjesečni trošak: €{monthly['total_monthly_cost']:.2f}
   - Trošak po compute satu: €{monthly['cost_per_compute_hour']:.4f}

WORKLOAD BREAKDOWN
"""
        
        for wl in monthly['workload_breakdown']:
            summary += f"""
   {wl['name']}:
      - Mjesečno izvršenja: {wl['monthly_runs']}
      - Sati po izvršenju: {wl['hours_per_run']:.2f}
      - Ukupno sati: {wl['total_monthly_hours']:.1f}
      - Efikasnost: {wl['efficiency']*100:.1f}%
      - Mjesečni trošak: €{wl['monthly_cost']:.2f}
"""
        
        # Scaling study za prvi workload
        if workloads:
            scaling = self.run_scaling_study(workloads[0])
            summary += f"""
SCALING STUDIJA: {scaling['workload']}
   - Optimalni broj jezgara: {scaling['optimal_cores']}
   - Maksimalni speedup: {scaling['max_speedup']:.2f}x
   - Efikasnost na max jezgara: {scaling['efficiency_at_max']*100:.1f}%
   
   Cores  |  Runtime (h)  |  Speedup  |  Efficiency
   -------|---------------|-----------|-------------
"""
            for r in scaling['scaling_results'][:8]:
                summary += f"   {r['cores']:5d}  |  {r['runtime_hours']:11.2f}  |  {r['speedup']:7.2f}x  |  {r['efficiency']*100:8.1f}%\n"
        
        summary += """
═══════════════════════════════════════════════════════════════
"""
        return summary
