"""
Modul za generisanje preporuka
===============================
Logika odlučivanja koja na osnovu prikupljenih podataka
i izračunatih score-ova generiše konkretne preporuke.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from config import MATURITY_LEVELS, MATURITY_THRESHOLDS


class RecommendationPriority(Enum):
    """Prioritet preporuke."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


class MigrationReadiness(Enum):
    """Spremnost za migraciju."""
    NOT_READY = "not_ready"
    NEEDS_PREPARATION = "needs_preparation"
    READY_WITH_SUPPORT = "ready_with_support"
    FULLY_READY = "fully_ready"


@dataclass
class Recommendation:
    """Strukturirana preporuka."""
    id: str
    title: str
    description: str
    priority: RecommendationPriority
    category: str  # infrastructure, team, software, process
    estimated_effort: str  # low, medium, high
    estimated_cost: str  # low, medium, high
    expected_impact: str
    prerequisites: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


class RecommendationEngine:
    """
    Engine za generisanje preporuka za HPC migraciju.
    Koristi rule-based sistem sa pragovima i prioritetima.
    """
    
    def __init__(self):
        self.recommendations = []
        self.maturity_score = 0
        self.maturity_level = 1
    
    def generate_recommendations(self, 
                                 infrastructure_results: Dict,
                                 team_results: Dict,
                                 software_results: Dict,
                                 simulation_results: Optional[Dict] = None) -> Dict:
        """
        Generiši kompletne preporuke na osnovu svih analiza.
        """
        self.recommendations = []
        
        # Izračunaj ukupni maturity score
        self.maturity_score = self._calculate_maturity_score(
            infrastructure_results, team_results, software_results
        )
        self.maturity_level = self._determine_maturity_level(self.maturity_score)
        
        # Generiši preporuke po kategorijama
        self._generate_infrastructure_recommendations(infrastructure_results)
        self._generate_team_recommendations(team_results)
        self._generate_software_recommendations(software_results)
        self._generate_process_recommendations(
            infrastructure_results, team_results, software_results
        )
        
        # Odredi spremnost za migraciju
        migration_readiness = self._assess_migration_readiness()
        
        # Kreiraj migration roadmap
        roadmap = self._create_migration_roadmap()
        
        # Sortiraj preporuke po prioritetu
        self.recommendations.sort(key=lambda x: x.priority.value)
        
        return {
            "maturity_score": self.maturity_score,
            "maturity_level": self.maturity_level,
            "maturity_level_name": MATURITY_LEVELS[self.maturity_level]["name"],
            "maturity_description": MATURITY_LEVELS[self.maturity_level]["description"],
            "migration_readiness": migration_readiness.value,
            "total_recommendations": len(self.recommendations),
            "recommendations_by_priority": self._group_by_priority(),
            "recommendations_by_category": self._group_by_category(),
            "migration_roadmap": roadmap,
            "quick_wins": self._identify_quick_wins(),
            "critical_blockers": self._identify_blockers()
        }
    
    def _calculate_maturity_score(self, infra: Dict, team: Dict, software: Dict) -> float:
        """Izračunaj ukupni maturity score."""
        # Težine komponenti
        weights = {
            "infrastructure": 0.30,
            "team": 0.30,
            "software": 0.25,
            "workload": 0.15
        }
        
        scores = {
            "infrastructure": infra.get("infrastructure_score", 0),
            "team": team.get("team_score", 0),
            "software": software.get("software_score", 0),
            "workload": infra.get("workload_analysis", {}).get("hpc_suitable_percentage", 50)
        }
        
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
        return weighted_sum
    
    def _determine_maturity_level(self, score: float) -> int:
        """Odredi nivo zrelosti na osnovu score-a."""
        for level, (low, high) in MATURITY_THRESHOLDS.items():
            if low <= score < high:
                return level
        return 5 if score >= 80 else 1
    
    def _generate_infrastructure_recommendations(self, results: Dict):
        """Generiši preporuke za infrastrukturu."""
        hpc_readiness = results.get("hpc_readiness", {})
        bottlenecks = results.get("bottlenecks", [])
        
        # Job scheduler
        if hpc_readiness.get("component_scores", {}).get("scheduler", 0) < 10:
            self.recommendations.append(Recommendation(
                id="INFRA-001",
                title="Implementacija Job Scheduler-a",
                description="Nedostaje job scheduler za upravljanje resursima i poslovima.",
                priority=RecommendationPriority.CRITICAL,
                category="infrastructure",
                estimated_effort="medium",
                estimated_cost="low",
                expected_impact="Omogućava efikasno dijeljenje resursa i scheduling poslova",
                action_items=[
                    "Instalirati SLURM job scheduler",
                    "Konfigurirati particije i QoS politike",
                    "Obučiti korisnike za korištenje sbatch/srun",
                    "Postaviti monitoring (Grafana/Prometheus)"
                ]
            ))
        
        # Shared filesystem
        if hpc_readiness.get("component_scores", {}).get("filesystem", 0) < 15:
            self.recommendations.append(Recommendation(
                id="INFRA-002",
                title="Implementacija Dijeljenog Filesystem-a",
                description="Lokalni storage ograničava skalabilnost i dijeljenje podataka.",
                priority=RecommendationPriority.HIGH,
                category="infrastructure",
                estimated_effort="high",
                estimated_cost="medium",
                expected_impact="Omogućava efikasan pristup podacima sa svih čvorova",
                action_items=[
                    "Evaluirati opcije: Lustre, BeeGFS, GPFS",
                    "Planirati storage kapacitet i IOPS potrebe",
                    "Implementirati tiered storage (SSD + HDD)",
                    "Konfigurirati backup i disaster recovery"
                ]
            ))
        
        # Interconnect
        if hpc_readiness.get("component_scores", {}).get("interconnect", 0) < 15:
            self.recommendations.append(Recommendation(
                id="INFRA-003",
                title="Nadogradnja Network Interconnect-a",
                description="Ethernet interconnect ima visoku latenciju za MPI workload-e.",
                priority=RecommendationPriority.MEDIUM,
                category="infrastructure",
                estimated_effort="high",
                estimated_cost="high",
                expected_impact="Smanjenje latencije sa ~50μs na <1μs za MPI",
                prerequisites=["Evaluacija workload komunikacionih pattern-a"],
                action_items=[
                    "Provesti benchmark MPI latencije",
                    "Evaluirati InfiniBand HDR/NDR opcije",
                    "Planirati fabric topologiju (Fat-tree, Dragonfly)",
                    "Razmotriti RoCE kao cost-effective alternativu"
                ]
            ))
        
        # Bottleneck specifične preporuke
        for bn in bottlenecks:
            if bn["type"] == "memory" and bn["severity"] == "high":
                self.recommendations.append(Recommendation(
                    id="INFRA-004",
                    title="Rješavanje Memory Bottleneck-a",
                    description=bn["description"],
                    priority=RecommendationPriority.HIGH,
                    category="infrastructure",
                    estimated_effort="medium",
                    estimated_cost="medium",
                    expected_impact="Eliminacija swapping-a i poboljšanje performansi",
                    action_items=[
                        "Profilisati memory usage po aplikacijama",
                        "Nadograditi RAM na kritičnim čvorovima",
                        "Implementirati memory-aware scheduling",
                        "Optimizovati aplikacije za manji footprint"
                    ]
                ))
    
    def _generate_team_recommendations(self, results: Dict):
        """Generiši preporuke za tim."""
        skill_assessment = results.get("skill_assessment", {})
        knowledge_gaps = results.get("knowledge_gaps", [])
        training_recs = results.get("training_recommendations", [])
        process_maturity = results.get("process_maturity", {})
        
        # Kritični skill gaps
        critical_gaps = [g for g in knowledge_gaps if g["severity"] == "critical"]
        if critical_gaps:
            skills_needed = ", ".join(set(g["skill"] for g in critical_gaps[:5]))
            self.recommendations.append(Recommendation(
                id="TEAM-001",
                title="Kritična Obuka iz Paralelnog Programiranja",
                description=f"Tim ima kritične nedostatke u: {skills_needed}",
                priority=RecommendationPriority.CRITICAL,
                category="team",
                estimated_effort="high",
                estimated_cost="medium",
                expected_impact="Omogućavanje efikasnog korištenja HPC resursa",
                action_items=[
                    "Organizovati intenzivnu MPI/OpenMP obuku (40h)",
                    "Registrovati tim za PRACE/ENCCS treninge",
                    "Uspostaviti mentorski program sa HPC ekspertima",
                    "Kreirati hands-on vježbe na realnim problemima"
                ]
            ))
        
        # Process gaps
        component_scores = process_maturity.get("component_scores", {})
        
        if component_scores.get("training_program", 0) == 0:
            self.recommendations.append(Recommendation(
                id="TEAM-002",
                title="Uspostavljanje HPC Training Programa",
                description="Nedostaje strukturirani program obuke za HPC.",
                priority=RecommendationPriority.HIGH,
                category="team",
                estimated_effort="medium",
                estimated_cost="low",
                expected_impact="Kontinuirano unapređenje vještina tima",
                action_items=[
                    "Kreirati curriculum za različite nivoe",
                    "Uspostaviti kvartalne radionice",
                    "Dokumentovati best practices",
                    "Mjeriti napredak kroz assessmente"
                ]
            ))
        
        if component_scores.get("code_review", 0) == 0:
            self.recommendations.append(Recommendation(
                id="TEAM-003",
                title="Implementacija Code Review Procesa",
                description="Nedostaje code review proces za HPC kod.",
                priority=RecommendationPriority.MEDIUM,
                category="team",
                estimated_effort="low",
                estimated_cost="low",
                expected_impact="Poboljšanje kvaliteta koda i transfer znanja",
                action_items=[
                    "Definisati code review checklist za HPC",
                    "Integrisati review u Git workflow",
                    "Fokusirati se na paralelizaciju i performanse",
                    "Koristiti automatske lintere (cppcheck, pylint)"
                ]
            ))
        
        # Dodaj preporuke iz team evaluator-a
        for tr in training_recs[:3]:
            self.recommendations.append(Recommendation(
                id=f"TEAM-T{tr['priority']}",
                title=tr["title"],
                description=tr["description"],
                priority=RecommendationPriority(min(tr["priority"] + 1, 5)),
                category="team",
                estimated_effort="medium",
                estimated_cost="medium",
                expected_impact=f"Adresira: {', '.join(tr['addresses_gaps'][:3])}",
                action_items=[
                    f"Trajanje: {tr['duration']}",
                    f"Provider: {tr['suggested_provider']}"
                ]
            ))
    
    def _generate_software_recommendations(self, results: Dict):
        """Generiši preporuke za softver."""
        parallelization = results.get("parallelization_assessment", {})
        antipatterns = results.get("antipatterns_detected", [])
        opportunities = results.get("optimization_opportunities", [])
        
        # Niska paralelizacija
        if parallelization.get("score", 0) < 40:
            self.recommendations.append(Recommendation(
                id="SW-001",
                title="Povećanje Nivoa Paralelizacije",
                description="Softver ima nizak nivo paralelizacije za HPC.",
                priority=RecommendationPriority.CRITICAL,
                category="software",
                estimated_effort="high",
                estimated_cost="low",
                expected_impact="Omogućavanje skaliranja na više jezgara/čvorova",
                action_items=[
                    "Profilisati aplikaciju za identifikaciju hotspots-a",
                    "Implementirati OpenMP za shared-memory paralelizaciju",
                    "Razmotriti MPI za distribuiranu paralelizaciju",
                    "Vektorizovati numeričke operacije"
                ]
            ))
        
        # Anti-patterns
        critical_aps = [ap for ap in antipatterns if ap.get("penalty", 0) > 15]
        if critical_aps:
            self.recommendations.append(Recommendation(
                id="SW-002",
                title="Refaktorisanje Anti-Pattern-a",
                description=f"Detektovano {len(critical_aps)} kritičnih anti-pattern-a.",
                priority=RecommendationPriority.HIGH,
                category="software",
                estimated_effort="medium",
                estimated_cost="low",
                expected_impact="Uklanjanje barijera za paralelizaciju",
                action_items=[
                    f"Riješiti: {ap['description']}" for ap in critical_aps[:4]
                ]
            ))
        
        # GPU akceleracija
        if not parallelization.get("uses_cuda", False):
            gpu_opp = [o for o in opportunities if o["type"] == "gpu_offloading"]
            if gpu_opp:
                self.recommendations.append(Recommendation(
                    id="SW-003",
                    title="Implementacija GPU Akceleracije",
                    description="Potencijal za značajno ubrzanje korištenjem GPU-a.",
                    priority=RecommendationPriority.MEDIUM,
                    category="software",
                    estimated_effort="high",
                    estimated_cost="low",
                    expected_impact="10-100x ubrzanje za compute-intensive operacije",
                    action_items=[
                        "Identifikovati GPU-pogodne kernele",
                        "Evaluirati CuPy/Numba za Python ili CUDA C++",
                        "Implementirati data transfer optimizacije",
                        "Benchmarkirati CPU vs GPU performanse"
                    ]
                ))
        
        # Profilisanje
        if not results.get("has_profiling_data", False):
            self.recommendations.append(Recommendation(
                id="SW-004",
                title="Profilisanje Aplikacija",
                description="Nedostaju podaci o profilisanju za informisanu optimizaciju.",
                priority=RecommendationPriority.HIGH,
                category="software",
                estimated_effort="low",
                estimated_cost="low",
                expected_impact="Identifikacija pravih bottleneck-a i hotspots-a",
                action_items=[
                    "Koristiti Intel VTune za CPU profilisanje",
                    "Koristiti NVIDIA Nsight za GPU profilisanje",
                    "Analizirati memory access patterns",
                    "Dokumentovati baseline performanse"
                ]
            ))
    
    def _generate_process_recommendations(self, infra: Dict, team: Dict, software: Dict):
        """Generiši preporuke za procese."""
        
        # CI/CD za HPC
        self.recommendations.append(Recommendation(
            id="PROC-001",
            title="Uspostavljanje HPC CI/CD Pipeline-a",
            description="Automatizovati testiranje i deployment HPC aplikacija.",
            priority=RecommendationPriority.MEDIUM,
            category="process",
            estimated_effort="medium",
            estimated_cost="low",
            expected_impact="Brži development cycle i pouzdaniji releases",
            action_items=[
                "Postaviti GitLab/GitHub Actions za CI",
                "Implementirati automatske unit i integration testove",
                "Dodati performance regression testove",
                "Automatizovati deployment na HPC klaster"
            ]
        ))
        
        # Monitoring
        if self.maturity_level < 4:
            self.recommendations.append(Recommendation(
                id="PROC-002",
                title="Implementacija HPC Monitoring-a",
                description="Uspostaviti sveobuhvatan monitoring HPC resursa.",
                priority=RecommendationPriority.MEDIUM,
                category="process",
                estimated_effort="medium",
                estimated_cost="low",
                expected_impact="Proaktivno upravljanje resursima i troubleshooting",
                action_items=[
                    "Instalirati Prometheus + Grafana stack",
                    "Konfigurirati SLURM metrics exporter",
                    "Postaviti alerte za kritične metrike",
                    "Kreirati dashboards za različite stakeholder-e"
                ]
            ))
    
    def _assess_migration_readiness(self) -> MigrationReadiness:
        """Procijeni spremnost za HPC migraciju."""
        critical_count = len([r for r in self.recommendations 
                            if r.priority == RecommendationPriority.CRITICAL])
        high_count = len([r for r in self.recommendations 
                         if r.priority == RecommendationPriority.HIGH])
        
        if critical_count > 2 or self.maturity_score < 25:
            return MigrationReadiness.NOT_READY
        elif critical_count > 0 or self.maturity_score < 45:
            return MigrationReadiness.NEEDS_PREPARATION
        elif high_count > 3 or self.maturity_score < 65:
            return MigrationReadiness.READY_WITH_SUPPORT
        else:
            return MigrationReadiness.FULLY_READY
    
    def generate_maturity_map(self, infra_score: float, team_score: float, software_score: float, data_interop_score: float = 0) -> str:
        """Generiši ASCII vizuelnu maturity mapu."""
        
        score = self.maturity_score
        level = self.maturity_level
        
        # Progress bar za ukupni score
        filled = int(score / 5)  # 20 karaktera = 100%
        empty = 20 - filled
        progress_bar = "█" * filled + "░" * empty
        
        # Nivo indikatori
        level_markers = []
        for i in range(1, 6):
            if i < level:
                level_markers.append("●")  # Prošli nivoi
            elif i == level:
                level_markers.append("◆")  # Trenutni nivo
            else:
                level_markers.append("○")  # Budući nivoi
        
        # Pozicija markera na liniji (svaki nivo = 10 karaktera)
        position = int((score / 100) * 50)
        position_line = " " * position + "▲"
        
        # Component progress bars
        def component_bar(score: float, width: int = 20) -> str:
            filled = int(score / 100 * width)
            return "█" * filled + "░" * (width - filled)
        
        # Maturity nivo opisi
        level_names = {
            1: "Početni",
            2: "Osnovni", 
            3: "Definisani",
            4: "Upravljani",
            5: "Optimizovani"
        }
        
        # ASCII art mapa
        maturity_map = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HPC MATURITY MAPA                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  UKUPNI SCORE: [{progress_bar}] {score:.1f}/100                     ║
║                                                                              ║
║  NIVO ZRELOSTI: {level} - {level_names[level]:<12}                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Nivo 1      Nivo 2      Nivo 3      Nivo 4      Nivo 5                    ║
║    Početni     Osnovni     Definisani  Upravljani  Optimizovani              ║
║       │           │           │           │           │                      ║
║       {level_markers[0]}───────────{level_markers[1]}───────────{level_markers[2]}───────────{level_markers[3]}───────────{level_markers[4]}                      ║
║  {position_line:<53}                      ║
║                    VI STE OVDJE                                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KOMPONENTE:                                                                 ║
║                                                                              ║
║    Infrastructure   [{component_bar(infra_score)}] {infra_score:5.1f}%               ║
║    Team Readiness   [{component_bar(team_score)}] {team_score:5.1f}%               ║
║    Software         [{component_bar(software_score)}] {software_score:5.1f}%               ║
║    Data & Interop   [{component_bar(data_interop_score)}] {data_interop_score:5.1f}%               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LEGENDA:                                                                    ║
║    ● Dostignuti nivo   ◆ Trenutni nivo   ○ Ciljni nivo                       ║
║    █ Postignuto        ░ Preostalo                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return maturity_map

    def _create_migration_roadmap(self) -> Dict:
        """Kreiraj roadmap za migraciju."""
        phases = []
        
        # Faza 1: Temelji (0-3 mjeseca)
        phase1_items = [r for r in self.recommendations 
                       if r.priority in [RecommendationPriority.CRITICAL]]
        phases.append({
            "phase": 1,
            "name": "Temelji",
            "duration": "0-3 mjeseca",
            "focus": "Rješavanje kritičnih nedostataka",
            "items": [{"id": r.id, "title": r.title} for r in phase1_items],
            "success_criteria": [
                "Job scheduler funkcionalan",
                "Tim ima osnovne HPC vještine",
                "Barem jedna aplikacija paralelizovana"
            ]
        })
        
        # Faza 2: Izgradnja (3-6 mjeseci)
        phase2_items = [r for r in self.recommendations 
                       if r.priority == RecommendationPriority.HIGH]
        phases.append({
            "phase": 2,
            "name": "Izgradnja",
            "duration": "3-6 mjeseci",
            "focus": "Optimizacija i proširenje",
            "items": [{"id": r.id, "title": r.title} for r in phase2_items],
            "success_criteria": [
                "Shared filesystem implementiran",
                "Code review proces aktivan",
                "Profilisanje završeno za ključne aplikacije"
            ]
        })
        
        # Faza 3: Optimizacija (6-12 mjeseci)
        phase3_items = [r for r in self.recommendations 
                       if r.priority in [RecommendationPriority.MEDIUM, 
                                        RecommendationPriority.LOW]]
        phases.append({
            "phase": 3,
            "name": "Optimizacija",
            "duration": "6-12 mjeseci",
            "focus": "Fino podešavanje i skaliranje",
            "items": [{"id": r.id, "title": r.title} for r in phase3_items[:5]],
            "success_criteria": [
                "Paralelna efikasnost >70%",
                "GPU akceleracija implementirana",
                "Monitoring i alerting aktivni"
            ]
        })
        
        return {
            "total_phases": len(phases),
            "estimated_total_duration": "12 mjeseci",
            "phases": phases
        }
    
    def _identify_quick_wins(self) -> List[Dict]:
        """Identifikuj brze pobjede (low effort, high impact)."""
        quick_wins = []
        
        for r in self.recommendations:
            if r.estimated_effort == "low" and r.priority.value <= 3:
                quick_wins.append({
                    "id": r.id,
                    "title": r.title,
                    "expected_impact": r.expected_impact,
                    "first_action": r.action_items[0] if r.action_items else ""
                })
        
        return quick_wins[:5]
    
    def _identify_blockers(self) -> List[Dict]:
        """Identifikuj kritične blokere."""
        blockers = []
        
        for r in self.recommendations:
            if r.priority == RecommendationPriority.CRITICAL:
                blockers.append({
                    "id": r.id,
                    "title": r.title,
                    "description": r.description,
                    "category": r.category
                })
        
        return blockers
    
    def _group_by_priority(self) -> Dict[str, List[Dict]]:
        """Grupiši preporuke po prioritetu."""
        groups = {}
        for priority in RecommendationPriority:
            recs = [r for r in self.recommendations if r.priority == priority]
            if recs:
                groups[priority.name] = [{
                    "id": r.id,
                    "title": r.title,
                    "category": r.category
                } for r in recs]
        return groups
    
    def _group_by_category(self) -> Dict[str, List[Dict]]:
        """Grupiši preporuke po kategoriji."""
        categories = ["infrastructure", "team", "software", "process"]
        groups = {}
        
        for cat in categories:
            recs = [r for r in self.recommendations if r.category == cat]
            if recs:
                groups[cat] = [{
                    "id": r.id,
                    "title": r.title,
                    "priority": r.priority.name
                } for r in recs]
        
        return groups
    
    def get_summary(self) -> str:
        """Generiši tekstualni sažetak preporuka."""
        migration_readiness = self._assess_migration_readiness()
        roadmap = self._create_migration_roadmap()
        
        summary = f"""
═══════════════════════════════════════════════════════════════
              PREPORUKE ZA HPC MIGRACIJU
═══════════════════════════════════════════════════════════════

MATURITY ASSESSMENT
   - Score: {self.maturity_score:.1f}/100
   - Nivo: {self.maturity_level} - {MATURITY_LEVELS[self.maturity_level]['name']}
   - Opis: {MATURITY_LEVELS[self.maturity_level]['description']}

SPREMNOST ZA MIGRACIJU: {migration_readiness.value.upper().replace('_', ' ')}

PREGLED PREPORUKA
   - Ukupno: {len(self.recommendations)}
   - Kritične: {len([r for r in self.recommendations if r.priority == RecommendationPriority.CRITICAL])}
   - Visoki prioritet: {len([r for r in self.recommendations if r.priority == RecommendationPriority.HIGH])}
   - Srednji prioritet: {len([r for r in self.recommendations if r.priority == RecommendationPriority.MEDIUM])}

KRITIČNI BLOKERI
"""
        
        for blocker in self._identify_blockers():
            summary += f"   - [{blocker['id']}] {blocker['title']}\n"
        
        summary += """
QUICK WINS (Low Effort, High Impact)
"""
        for qw in self._identify_quick_wins():
            summary += f"   - [{qw['id']}] {qw['title']}\n"
        
        summary += f"""
 MIGRATION ROADMAP

"""
        for phase in roadmap["phases"]:
            summary += f"""   FAZA {phase['phase']}: {phase['name']} ({phase['duration']})
   Focus: {phase['focus']}
   Items: {len(phase['items'])} preporuka
   
"""
        
        summary += """
═══════════════════════════════════════════════════════════════
            DETALJNE PREPORUKE PO KATEGORIJI
═══════════════════════════════════════════════════════════════
"""
        
        for category, recs in self._group_by_category().items():
            summary += f"\n{category.upper()}\n"
            for rec in recs:
                full_rec = next((r for r in self.recommendations if r.id == rec["id"]), None)
                if full_rec:
                    summary += f"""
   [{rec['id']}] {rec['title']}
   Priority: {rec['priority']} | Effort: {full_rec.estimated_effort} | Cost: {full_rec.estimated_cost}
   → {full_rec.description}
   Action items:
"""
                    for action in full_rec.action_items[:3]:
                        summary += f"      - {action}\n"
        
        summary += """
═══════════════════════════════════════════════════════════════
"""
        return summary
