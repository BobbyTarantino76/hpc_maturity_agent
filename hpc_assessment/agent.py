"""
HPC Maturity Assessment Agent
==============================
Glavni modul AI agenta za procjenu zrelosti organizacija
za korištenje HPC infrastrukture.

Autor: AI Agent
Verzija: 1.0
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from config import MATURITY_LEVELS, ASSESSMENT_WEIGHTS
from infrastructure_analyzer import (
    InfrastructureProfile, InfrastructureAnalyzer,
    ComputeNode, Workload, ResourceMetrics
)
from team_evaluator import (
    TeamProfile, TeamMember, TeamReadinessEvaluator, SkillLevel
)
from software_analyzer import (
    SoftwareProfile, SoftwareAnalyzer
)
from simulator import (
    HPCSimulator, HPCClusterConfig, SimulationWorkload
)
from recommendation_engine import RecommendationEngine
from data_interoperability_analyzer import (
    DataInteroperabilityProfile, DataInteroperabilityAnalyzer,
    DataSource, SystemInterface, DataAccessLevel, DataFormat,
    create_demo_data_interop_profile
)


@dataclass
class AssessmentInput:
    """Ulazni podaci za procjenu."""
    organization_name: str
    infrastructure: Optional[InfrastructureProfile] = None
    team: Optional[TeamProfile] = None
    software: Optional[SoftwareProfile] = None
    target_cluster: Optional[HPCClusterConfig] = None
    data_interoperability: Optional[DataInteroperabilityProfile] = None


@dataclass
class AssessmentOutput:
    """Izlazni podaci procjene."""
    timestamp: str
    organization_name: str
    maturity_score: float
    maturity_level: int
    maturity_level_name: str
    infrastructure_score: float
    team_score: float
    software_score: float
    data_interop_score: float
    migration_readiness: str
    summary: str
    detailed_results: Dict
    recommendations: List[Dict]
    simulation_results: Optional[Dict]


class HPCMaturityAgent:
    """
    AI Agent za procjenu zrelosti za HPC.
    
    Funkcionalnosti:
    1. Analiza infrastrukture i workload-a
    2. Evaluacija spremnosti tima
    3. Procjena optimizacije softvera
    4. Generisanje preporuka za migraciju
    5. Simulacija kapaciteta i resursa
    """
    
    def __init__(self):
        self.infrastructure_analyzer = None
        self.team_evaluator = None
        self.software_analyzer = None
        self.simulator = None
        self.data_interop_analyzer = None
        self.recommendation_engine = RecommendationEngine()
        
        self.results = {}
        self.assessment_complete = False
    
    def run_assessment(self, input_data: AssessmentInput) -> AssessmentOutput:
        """
        Izvrši kompletnu procjenu zrelosti.
        
        Args:
            input_data: AssessmentInput sa svim potrebnim podacima
            
        Returns:
            AssessmentOutput sa rezultatima i preporukama
        """
        print(f"\n{'='*60}")
        print(f"  HPC MATURITY ASSESSMENT: {input_data.organization_name}")
        print(f"{'='*60}\n")
        
        # 1. Analiza infrastrukture
        print("Analiziram infrastrukturu...")
        infra_results = self._analyze_infrastructure(input_data.infrastructure)
        
        # 2. Evaluacija tima
        print("Evaluiram spremnost tima...")
        team_results = self._evaluate_team(input_data.team)
        
        # 3. Analiza softvera
        print("Analiziram softver...")
        software_results = self._analyze_software(input_data.software)
        
        # 4. Simulacija (ako je definisan target klaster)
        print("Pokrećem simulaciju...")
        simulation_results = None
        if input_data.target_cluster and input_data.infrastructure:
            simulation_results = self._run_simulation(
                input_data.target_cluster,
                input_data.infrastructure.workloads
            )
        
        # 5. Analiza dostupnosti podataka i interoperabilnosti
        print("Analiziram dostupnost podataka i interoperabilnost...")
        data_interop_results = self._analyze_data_interoperability(input_data.data_interoperability)
        
        # 6. Generisanje preporuka
        print("Generišem preporuke...")
        recommendations = self.recommendation_engine.generate_recommendations(
            infra_results, team_results, software_results, simulation_results
        )
        
        # Dodaj data interop preporuke
        if data_interop_results.get("recommendations"):
            for rec in data_interop_results["recommendations"]:
                recommendations["recommendations_by_priority"].setdefault(rec["priority"], []).append(rec)
        
        # Kompajliraj rezultate
        self.results = {
            "infrastructure": infra_results,
            "team": team_results,
            "software": software_results,
            "simulation": simulation_results,
            "data_interoperability": data_interop_results,
            "recommendations": recommendations
        }
        
        self.assessment_complete = True
        
        # Kreiraj output
        output = AssessmentOutput(
            timestamp=datetime.now().isoformat(),
            organization_name=input_data.organization_name,
            maturity_score=recommendations["maturity_score"],
            maturity_level=recommendations["maturity_level"],
            maturity_level_name=recommendations["maturity_level_name"],
            infrastructure_score=infra_results.get("infrastructure_score", 0),
            team_score=team_results.get("team_score", 0),
            software_score=software_results.get("software_score", 0),
            data_interop_score=data_interop_results.get("data_interop_score", 0),
            migration_readiness=recommendations["migration_readiness"],
            summary=self._generate_executive_summary(),
            detailed_results=self.results,
            recommendations=recommendations["recommendations_by_priority"],
            simulation_results=simulation_results
        )
        
        print("\nProcjena završena!\n")
        
        return output
    
    def _analyze_infrastructure(self, profile: Optional[InfrastructureProfile]) -> Dict:
        """Analiziraj infrastrukturu."""
        if profile is None:
            return {"status": "no_data", "infrastructure_score": 0}
        
        self.infrastructure_analyzer = InfrastructureAnalyzer(profile)
        return self.infrastructure_analyzer.analyze()
    
    def _evaluate_team(self, profile: Optional[TeamProfile]) -> Dict:
        """Evaluiraj tim."""
        if profile is None:
            return {"status": "no_data", "team_score": 0}
        
        self.team_evaluator = TeamReadinessEvaluator(profile)
        return self.team_evaluator.evaluate()
    
    def _analyze_software(self, profile: Optional[SoftwareProfile]) -> Dict:
        """Analiziraj softver."""
        if profile is None:
            return {"status": "no_data", "software_score": 0}
        
        self.software_analyzer = SoftwareAnalyzer(profile)
        return self.software_analyzer.analyze()
    
    def _run_simulation(self, cluster: HPCClusterConfig, 
                       workloads: List[Workload]) -> Dict:
        """Pokreni simulaciju."""
        self.simulator = HPCSimulator(cluster)
        
        # Konvertuj Workload u SimulationWorkload
        sim_workloads = []
        for wl in workloads:
            sim_wl = SimulationWorkload(
                name=wl.name,
                workload_type=wl.workload_type,
                base_runtime_hours=wl.avg_runtime_hours,
                cpu_cores_required=max(1, int(wl.avg_cpu_utilization / 10)),
                memory_gb_required=wl.data_size_gb * 0.5,
                parallelizable_fraction=wl.parallelizable_fraction,
                data_size_gb=wl.data_size_gb,
                monthly_runs=wl.frequency_per_month
            )
            sim_workloads.append(sim_wl)
        
        return self.simulator.simulate_monthly_usage(sim_workloads)
    
    def _analyze_data_interoperability(self, profile: Optional[DataInteroperabilityProfile]) -> Dict:
        """Analiziraj dostupnost podataka i interoperabilnost."""
        if profile is None:
            return {
                "status": "no_data", 
                "data_interop_score": 0,
                "data_availability": {},
                "data_quality": {},
                "interoperability": {},
                "integration_readiness": {},
                "recommendations": []
            }
        
        self.data_interop_analyzer = DataInteroperabilityAnalyzer(profile)
        return self.data_interop_analyzer.analyze()
    
    def _generate_executive_summary(self) -> str:
        """Generiši executive summary."""
        if not self.assessment_complete:
            return "Assessment not complete"
        
        r = self.results
        rec = r["recommendations"]
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           EXECUTIVE SUMMARY - HPC MATURITY ASSESSMENT        ║
╠══════════════════════════════════════════════════════════════╣

  UKUPNI MATURITY SCORE: {rec['maturity_score']:.1f}/100
  
  NIVO ZRELOSTI: {rec['maturity_level']} - {rec['maturity_level_name']}
  {rec['maturity_description']}

  SPREMNOST ZA MIGRACIJU: {rec['migration_readiness'].upper().replace('_', ' ')}

╠══════════════════════════════════════════════════════════════╣
  COMPONENT SCORES:
  
  ├── Infrastructure: {r['infrastructure'].get('infrastructure_score', 0):.1f}/100
  ├── Team Readiness: {r['team'].get('team_score', 0):.1f}/100
  ├── Software Optimization: {r['software'].get('software_score', 0):.1f}/100
  └── Data & Interoperability: {r['data_interoperability'].get('data_interop_score', 0):.1f}/100

╠══════════════════════════════════════════════════════════════╣
  KRITIČNI NALAZI:
  
"""
        # Dodaj kritične blokere
        for blocker in rec.get("critical_blockers", [])[:3]:
            summary += f"  {blocker['title']}\n"
        
        summary += """
╠══════════════════════════════════════════════════════════════╣
  PREPORUČENI SLJEDEĆI KORACI:
  
"""
        # Dodaj quick wins
        for qw in rec.get("quick_wins", [])[:3]:
            summary += f"  {qw['title']}\n"
        
        summary += """
╠══════════════════════════════════════════════════════════════╣
  PROCIJENJENI TIMELINE ZA MIGRACIJU:
  
"""
        roadmap = rec.get("migration_roadmap", {})
        for phase in roadmap.get("phases", []):
            summary += f"  Faza {phase['phase']}: {phase['name']} - {phase['duration']}\n"
        
        summary += f"""
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def get_maturity_map(self) -> str:
        """Generiši ASCII maturity mapu."""
        if not self.assessment_complete:
            return "Please run assessment first."
        
        r = self.results
        infra_score = r['infrastructure'].get('infrastructure_score', 0)
        team_score = r['team'].get('team_score', 0)
        software_score = r['software'].get('software_score', 0)
        data_interop_score = r['data_interoperability'].get('data_interop_score', 0)
        
        return self.recommendation_engine.generate_maturity_map(
            infra_score, team_score, software_score, data_interop_score
        )
    
    def get_detailed_report(self) -> str:
        """Generiši detaljan izvještaj."""
        if not self.assessment_complete:
            return "Please run assessment first."
        
        report = ""
        
        # Infrastructure analysis
        if self.infrastructure_analyzer:
            report += self.infrastructure_analyzer.get_summary()
        
        # Team evaluation
        if self.team_evaluator:
            report += self.team_evaluator.get_summary()
        
        # Software analysis
        if self.software_analyzer:
            report += self.software_analyzer.get_summary()
        
        # Data interoperability analysis
        if hasattr(self, 'data_interop_analyzer') and self.data_interop_analyzer:
            report += self.data_interop_analyzer.generate_report()
        
        # Simulation results
        if self.simulator and self.results.get("simulation"):
            workloads = []
            for wl in self.results["simulation"].get("workload_breakdown", []):
                sim_wl = SimulationWorkload(
                    name=wl["name"],
                    workload_type="balanced",
                    base_runtime_hours=wl["hours_per_run"],
                    cpu_cores_required=8,
                    memory_gb_required=16,
                    monthly_runs=wl["monthly_runs"]
                )
                workloads.append(sim_wl)
            if workloads:
                report += self.simulator.get_summary(workloads)
        
        # Recommendations
        report += self.recommendation_engine.get_summary()
        
        return report
    
    def export_results(self, format: str = "json") -> str:
        """Eksportuj rezultate u željenom formatu."""
        if not self.assessment_complete:
            return "{}"
        
        if format == "json":
            # Konvertuj u JSON-serializable format
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "maturity_score": self.results["recommendations"]["maturity_score"],
                "maturity_level": self.results["recommendations"]["maturity_level"],
                "infrastructure_score": self.results["infrastructure"].get("infrastructure_score", 0),
                "team_score": self.results["team"].get("team_score", 0),
                "software_score": self.results["software"].get("software_score", 0),
                "migration_readiness": self.results["recommendations"]["migration_readiness"],
                "recommendations": self.results["recommendations"]["recommendations_by_priority"],
                "roadmap": self.results["recommendations"]["migration_roadmap"]
            }
            return json.dumps(export_data, indent=2, default=str)
        
        return self.get_detailed_report()


def create_demo_data() -> AssessmentInput:
    """Kreiraj demo podatke za testiranje."""
    
    # Demo infrastruktura
    nodes = [
        ComputeNode("node001", cpu_cores=32, cpu_frequency_ghz=2.5, 
                   memory_gb=128, gpu_count=2, gpu_memory_gb=16),
        ComputeNode("node002", cpu_cores=32, cpu_frequency_ghz=2.5,
                   memory_gb=128, gpu_count=2, gpu_memory_gb=16),
        ComputeNode("node003", cpu_cores=64, cpu_frequency_ghz=2.2,
                   memory_gb=256, gpu_count=0),
        ComputeNode("node004", cpu_cores=64, cpu_frequency_ghz=2.2,
                   memory_gb=256, gpu_count=0),
    ]
    
    workloads = [
        Workload("wl001", "CFD Simulacija", "compute_intensive",
                avg_cpu_utilization=85, avg_memory_utilization=60,
                avg_io_operations_per_sec=100, avg_runtime_hours=24,
                parallelizable_fraction=0.92, data_size_gb=500,
                frequency_per_month=4),
        Workload("wl002", "Data Analytics", "memory_intensive",
                avg_cpu_utilization=40, avg_memory_utilization=85,
                avg_io_operations_per_sec=1000, avg_runtime_hours=8,
                parallelizable_fraction=0.75, data_size_gb=2000,
                frequency_per_month=20),
        Workload("wl003", "ML Training", "gpu_accelerated",
                avg_cpu_utilization=30, avg_memory_utilization=70,
                avg_io_operations_per_sec=500, avg_runtime_hours=48,
                parallelizable_fraction=0.88, data_size_gb=100,
                frequency_per_month=8),
    ]
    
    metrics = ResourceMetrics(
        avg_cpu_utilization=55,
        peak_cpu_utilization=92,
        avg_memory_utilization=65,
        peak_memory_utilization=88,
        avg_gpu_utilization=45,
        idle_time_percentage=15
    )
    
    infrastructure = InfrastructureProfile(
        organization_name="Demo Organization",
        nodes=nodes,
        workloads=workloads,
        metrics=metrics,
        has_job_scheduler=False,
        scheduler_type="none",
        has_shared_filesystem=True,
        filesystem_type="nfs",
        interconnect_type="ethernet"
    )
    
    # Demo tim
    members = [
        TeamMember(
            "m001", "Ana Petrović", "Lead Developer", 
            years_experience=8,
            skills={
                "MPI": SkillLevel.INTERMEDIATE,
                "OpenMP": SkillLevel.ADVANCED,
                "CUDA": SkillLevel.BEGINNER,
                "slurm": SkillLevel.NONE,
                "profilers": SkillLevel.INTERMEDIATE,
                "version_control": SkillLevel.EXPERT
            },
            hpc_projects_completed=3
        ),
        TeamMember(
            "m002", "Marko Jovanović", "Software Engineer",
            years_experience=4,
            skills={
                "MPI": SkillLevel.BEGINNER,
                "OpenMP": SkillLevel.INTERMEDIATE,
                "threading": SkillLevel.INTERMEDIATE,
                "version_control": SkillLevel.ADVANCED,
                "containers": SkillLevel.INTERMEDIATE
            },
            hpc_projects_completed=1
        ),
        TeamMember(
            "m003", "Ivana Nikolić", "Data Scientist",
            years_experience=5,
            skills={
                "CUDA": SkillLevel.INTERMEDIATE,
                "numerical_methods": SkillLevel.ADVANCED,
                "data_analysis": SkillLevel.EXPERT,
                "version_control": SkillLevel.INTERMEDIATE
            },
            hpc_projects_completed=2
        ),
    ]
    
    team = TeamProfile(
        team_name="HPC Development Team",
        members=members,
        has_hpc_training_program=False,
        has_documentation=True,
        has_code_review_process=False,
        collaboration_tools=["Git", "Slack", "Confluence"]
    )
    
    # Demo softver
    demo_code = '''
import numpy as np
from multiprocessing import Pool

def compute_heavy(data):
    result = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            result[i][j] = np.sin(data[i][j]) * np.cos(data[i][j])
    return result

def main():
    data = np.random.rand(1000, 1000)
    
    # Sequential processing
    for iteration in range(100):
        result = compute_heavy(data)
        with open(f'output_{iteration}.dat', 'w') as f:
            f.write(str(result.sum()))
    
    # Some parallelization attempt
    with Pool(4) as p:
        chunks = np.array_split(data, 4)
        results = p.map(compute_heavy, chunks)

if __name__ == "__main__":
    main()
'''
    
    software = SoftwareProfile(
        name="SimulationApp",
        language="python",
        source_code=demo_code,
        estimated_loc=1500,
        uses_mpi=False,
        uses_openmp=False,
        uses_cuda=False,
        uses_vectorization=True,
        has_profiling_data=False,
        scalability_tested=False,
        max_tested_cores=4
    )
    
    # Target HPC klaster za simulaciju
    target_cluster = HPCClusterConfig(
        name="Target HPC Cluster",
        total_nodes=20,
        cores_per_node=64,
        memory_per_node_gb=256,
        gpus_per_node=4,
        interconnect_bandwidth_gbps=200,
        parallel_efficiency=0.85
    )
    
    # Demo data interoperability profil
    data_interop = create_demo_data_interop_profile()
    
    return AssessmentInput(
        organization_name="Demo Organization",
        infrastructure=infrastructure,
        team=team,
        software=software,
        target_cluster=target_cluster,
        data_interoperability=data_interop
    )


if __name__ == "__main__":
    # Demo izvršenje
    print("\n" + "="*60)
    print("   HPC MATURITY ASSESSMENT AGENT - DEMO")
    print("="*60)
    
    # Kreiraj agenta
    agent = HPCMaturityAgent()
    
    # Učitaj demo podatke
    demo_input = create_demo_data()
    
    # Pokreni procjenu
    output = agent.run_assessment(demo_input)
    
    # Prikaži maturity mapu
    print(agent.get_maturity_map())
    
    # Prikaži executive summary
    print(output.summary)
    
    # Prikaži detaljan izvještaj
    print("\n" + "="*60)
    print("   DETAILED REPORT")
    print("="*60)
    print(agent.get_detailed_report())
    
    # Eksportuj JSON
    print("\n" + "="*60)
    print("   JSON EXPORT")
    print("="*60)
    print(agent.export_results("json"))
