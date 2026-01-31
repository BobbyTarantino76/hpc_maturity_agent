"""
Modul za evaluaciju spremnosti timova
======================================
Strukturisani model procjene znanja i operativne spremnosti
timova za paralelno programiranje i HPC okruženja.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import statistics

from config import TEAM_SKILL_CATEGORIES


class SkillLevel(Enum):
    """Nivoi vještina."""
    NONE = 0
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class TeamMember:
    """Profil člana tima."""
    member_id: str
    name: str
    role: str
    years_experience: float
    skills: Dict[str, SkillLevel] = field(default_factory=dict)
    certifications: List[str] = field(default_factory=list)
    hpc_projects_completed: int = 0
    
    def get_skill_level(self, skill: str) -> int:
        """Dohvati numeričku vrijednost vještine."""
        return self.skills.get(skill, SkillLevel.NONE).value


@dataclass
class TeamProfile:
    """Profil tima."""
    team_name: str
    members: List[TeamMember] = field(default_factory=list)
    has_hpc_training_program: bool = False
    has_documentation: bool = False
    has_code_review_process: bool = False
    collaboration_tools: List[str] = field(default_factory=list)
    
    def team_size(self) -> int:
        return len(self.members)


class TeamReadinessEvaluator:
    """
    Evaluator spremnosti tima za HPC.
    Koristi upitnik-bazirani scoring model.
    """
    
    def __init__(self, team: TeamProfile):
        self.team = team
        self.evaluation_results = {}
    
    def evaluate(self) -> Dict:
        """Izvrši kompletnu evaluaciju tima."""
        self.evaluation_results = {
            "skill_assessment": self._assess_skills(),
            "experience_assessment": self._assess_experience(),
            "process_maturity": self._assess_processes(),
            "knowledge_gaps": self._identify_knowledge_gaps(),
            "training_recommendations": self._generate_training_recommendations(),
            "team_score": 0
        }
        
        self.evaluation_results["team_score"] = self._calculate_team_score()
        
        return self.evaluation_results
    
    def _assess_skills(self) -> Dict:
        """Procjena vještina po kategorijama."""
        if not self.team.members:
            return {"status": "no_members", "overall": 0}
        
        category_scores = {}
        
        for category, config in TEAM_SKILL_CATEGORIES.items():
            skills = config["skills"]
            weight = config["weight"]
            
            # Prosječna razina vještine u kategoriji
            category_skill_scores = []
            
            for skill in skills:
                member_scores = []
                for member in self.team.members:
                    score = member.get_skill_level(skill)
                    member_scores.append(score)
                
                if member_scores:
                    # Uzimamo maksimum jer je dovoljno da neko u timu ima vještinu
                    category_skill_scores.append(max(member_scores))
            
            if category_skill_scores:
                # Normalizuj na 0-100 skalu (max je 4 = EXPERT)
                avg_score = statistics.mean(category_skill_scores)
                normalized_score = (avg_score / 4) * 100
                category_scores[category] = {
                    "raw_score": avg_score,
                    "normalized_score": normalized_score,
                    "weight": weight,
                    "weighted_score": normalized_score * weight
                }
            else:
                category_scores[category] = {
                    "raw_score": 0,
                    "normalized_score": 0,
                    "weight": weight,
                    "weighted_score": 0
                }
        
        total_weighted = sum(c["weighted_score"] for c in category_scores.values())
        
        return {
            "category_scores": category_scores,
            "overall_skill_score": total_weighted
        }
    
    def _assess_experience(self) -> Dict:
        """Procjena iskustva tima."""
        if not self.team.members:
            return {"status": "no_members", "score": 0}
        
        # Prosječno iskustvo
        avg_experience = statistics.mean([m.years_experience for m in self.team.members])
        
        # HPC projekti
        total_hpc_projects = sum(m.hpc_projects_completed for m in self.team.members)
        
        # Sertifikati
        all_certs = []
        for m in self.team.members:
            all_certs.extend(m.certifications)
        unique_certs = list(set(all_certs))
        
        # HPC relevantni sertifikati
        hpc_relevant_certs = [c for c in unique_certs if any(
            keyword in c.lower() for keyword in ['hpc', 'cuda', 'parallel', 'mpi', 'cluster']
        )]
        
        # Experience score (0-100)
        exp_score = min(avg_experience * 10, 40)  # Max 40 za iskustvo
        project_score = min(total_hpc_projects * 5, 40)  # Max 40 za projekte
        cert_score = min(len(hpc_relevant_certs) * 10, 20)  # Max 20 za sertifikate
        
        total_score = exp_score + project_score + cert_score
        
        return {
            "avg_years_experience": avg_experience,
            "total_hpc_projects": total_hpc_projects,
            "unique_certifications": unique_certs,
            "hpc_relevant_certifications": hpc_relevant_certs,
            "experience_score": total_score
        }
    
    def _assess_processes(self) -> Dict:
        """Procjena zrelosti procesa."""
        scores = {}
        
        # Training program
        scores["training_program"] = 25 if self.team.has_hpc_training_program else 0
        
        # Dokumentacija
        scores["documentation"] = 25 if self.team.has_documentation else 0
        
        # Code review
        scores["code_review"] = 25 if self.team.has_code_review_process else 0
        
        # Collaboration tools
        essential_tools = ["git", "jira", "slack", "confluence"]
        tool_coverage = sum(1 for t in essential_tools 
                          if any(t in tool.lower() for tool in self.team.collaboration_tools))
        scores["collaboration"] = (tool_coverage / len(essential_tools)) * 25
        
        total_score = sum(scores.values())
        
        return {
            "component_scores": scores,
            "process_maturity_score": total_score
        }
    
    def _identify_knowledge_gaps(self) -> List[Dict]:
        """Identifikacija nedostataka u znanju."""
        gaps = []
        
        if not self.team.members:
            return [{"area": "team", "severity": "critical", "description": "Nema članova tima"}]
        
        # Provjeri svaku kategoriju vještina
        for category, config in TEAM_SKILL_CATEGORIES.items():
            skills = config["skills"]
            
            for skill in skills:
                # Pronađi maksimalni nivo vještine u timu
                max_level = max(
                    (member.get_skill_level(skill) for member in self.team.members),
                    default=0
                )
                
                if max_level == 0:
                    gaps.append({
                        "area": category,
                        "skill": skill,
                        "severity": "high",
                        "current_level": "Nema",
                        "target_level": "Intermediate",
                        "description": f"Niko u timu nema vještinu: {skill}"
                    })
                elif max_level == 1:
                    gaps.append({
                        "area": category,
                        "skill": skill,
                        "severity": "medium",
                        "current_level": "Beginner",
                        "target_level": "Intermediate",
                        "description": f"Samo početni nivo za: {skill}"
                    })
        
        # Kritični nedostaci za HPC
        critical_skills = ["MPI", "OpenMP", "slurm", "profilers"]
        for skill in critical_skills:
            max_level = max(
                (member.get_skill_level(skill) for member in self.team.members),
                default=0
            )
            if max_level < 2:  # Manje od Intermediate
                # Dodaj samo ako već nije dodat
                existing = any(g["skill"] == skill for g in gaps)
                if not existing:
                    gaps.append({
                        "area": "critical_hpc",
                        "skill": skill,
                        "severity": "critical",
                        "current_level": SkillLevel(max_level).name if max_level > 0 else "Nema",
                        "target_level": "Advanced",
                        "description": f"Kritična HPC vještina ispod potrebnog nivoa: {skill}"
                    })
        
        # Sortiraj po severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        return gaps
    
    def _generate_training_recommendations(self) -> List[Dict]:
        """Generiši preporuke za obuku."""
        recommendations = []
        
        gaps = self._identify_knowledge_gaps()
        
        # Grupiši po kategoriji
        critical_gaps = [g for g in gaps if g["severity"] == "critical"]
        high_gaps = [g for g in gaps if g["severity"] == "high"]
        
        # Kritične preporuke
        if any(g["skill"] in ["MPI", "OpenMP"] for g in critical_gaps):
            recommendations.append({
                "priority": 1,
                "type": "course",
                "title": "Parallel Programming Fundamentals",
                "description": "Obuka iz osnova paralelnog programiranja sa MPI i OpenMP",
                "duration": "40 sati",
                "suggested_provider": "PRACE Training / ENCCS",
                "addresses_gaps": ["MPI", "OpenMP", "threading"]
            })
        
        if any(g["skill"] == "CUDA" for g in gaps):
            recommendations.append({
                "priority": 2,
                "type": "course", 
                "title": "GPU Programming with CUDA",
                "description": "Obuka iz GPU programiranja sa NVIDIA CUDA",
                "duration": "24 sata",
                "suggested_provider": "NVIDIA DLI",
                "addresses_gaps": ["CUDA", "gpu_offloading"]
            })
        
        if any(g["skill"] == "slurm" for g in gaps):
            recommendations.append({
                "priority": 1,
                "type": "workshop",
                "title": "HPC Job Scheduling with SLURM",
                "description": "Praktična radionica za korištenje SLURM-a",
                "duration": "8 sati",
                "suggested_provider": "Internal / HPC Center",
                "addresses_gaps": ["slurm", "pbs"]
            })
        
        if any(g["skill"] == "profilers" for g in gaps):
            recommendations.append({
                "priority": 2,
                "type": "workshop",
                "title": "Performance Analysis and Profiling",
                "description": "Korištenje alata za profilisanje HPC aplikacija",
                "duration": "16 sati",
                "suggested_provider": "Intel / AMD / NVIDIA",
                "addresses_gaps": ["profilers", "optimization"]
            })
        
        # Procesne preporuke
        if not self.team.has_code_review_process:
            recommendations.append({
                "priority": 3,
                "type": "process",
                "title": "Implement Code Review Process",
                "description": "Uspostaviti proces code review-a za HPC kod",
                "duration": "Ongoing",
                "suggested_provider": "Internal",
                "addresses_gaps": ["code_quality", "knowledge_sharing"]
            })
        
        if not self.team.has_documentation:
            recommendations.append({
                "priority": 3,
                "type": "process",
                "title": "Documentation Standards",
                "description": "Uspostaviti standarde dokumentacije za HPC projekte",
                "duration": "Ongoing", 
                "suggested_provider": "Internal",
                "addresses_gaps": ["documentation", "maintainability"]
            })
        
        recommendations.sort(key=lambda x: x["priority"])
        
        return recommendations
    
    def _calculate_team_score(self) -> float:
        """Izračunaj ukupni team readiness score."""
        skill_score = self.evaluation_results["skill_assessment"].get("overall_skill_score", 0)
        experience_score = self.evaluation_results["experience_assessment"].get("experience_score", 0)
        process_score = self.evaluation_results["process_maturity"].get("process_maturity_score", 0)
        
        # Penalizacija za knowledge gaps
        gaps = self.evaluation_results["knowledge_gaps"]
        critical_gaps = len([g for g in gaps if g["severity"] == "critical"])
        high_gaps = len([g for g in gaps if g["severity"] == "high"])
        
        gap_penalty = critical_gaps * 5 + high_gaps * 2
        
        # Weighted score
        weighted_score = (
            skill_score * 0.4 +
            experience_score * 0.35 +
            process_score * 0.25
        )
        
        final_score = max(0, weighted_score - gap_penalty)
        
        return min(100, final_score)
    
    def get_summary(self) -> str:
        """Generiši tekstualni sažetak evaluacije."""
        if not self.evaluation_results:
            self.evaluate()
        
        r = self.evaluation_results
        
        summary = f"""
═══════════════════════════════════════════════════════════════
              EVALUACIJA TIMA: {self.team.team_name}
═══════════════════════════════════════════════════════════════

TIM
   - Veličina tima: {self.team.team_size()} članova
   - Prosječno iskustvo: {r['experience_assessment'].get('avg_years_experience', 0):.1f} godina
   - HPC projekti: {r['experience_assessment'].get('total_hpc_projects', 0)}

VJEŠTINE (Score: {r['skill_assessment'].get('overall_skill_score', 0):.1f}/100)
"""
        
        for cat, scores in r['skill_assessment'].get('category_scores', {}).items():
            summary += f"   - {cat}: {scores['normalized_score']:.1f}/100\n"
        
        summary += f"""
PROCESI (Score: {r['process_maturity'].get('process_maturity_score', 0):.1f}/100)
   - Training program: {'Da' if self.team.has_hpc_training_program else 'Ne'}
   - Dokumentacija: {'Da' if self.team.has_documentation else 'Ne'}
   - Code review: {'Da' if self.team.has_code_review_process else 'Ne'}

 KNOWLEDGE GAPS
"""
        
        critical_gaps = [g for g in r['knowledge_gaps'] if g['severity'] == 'critical']
        if critical_gaps:
            summary += "   KRITIČNI:\n"
            for gap in critical_gaps[:3]:
                summary += f"   - {gap['skill']}: {gap['description']}\n"
        
        summary += f"""
TOP PREPORUKE ZA OBUKU
"""
        for rec in r['training_recommendations'][:3]:
            summary += f"   {rec['priority']}. {rec['title']} ({rec['duration']})\n"
        
        summary += f"""
═══════════════════════════════════════════════════════════════
               TEAM READINESS SCORE: {r['team_score']:.1f}/100
═══════════════════════════════════════════════════════════════
"""
        return summary


# Upitnik za procjenu tima (može se koristiti za prikupljanje podataka)
TEAM_ASSESSMENT_QUESTIONNAIRE = {
    "parallel_programming": [
        {
            "id": "q1",
            "question": "Da li tim ima iskustva sa MPI programiranjem?",
            "options": [
                (0, "Nema iskustva"),
                (1, "Osnovno razumijevanje"),
                (2, "Može pisati jednostavne MPI programe"),
                (3, "Može optimizovati MPI komunikaciju"),
                (4, "Ekspertski nivo, skalabilne aplikacije")
            ],
            "maps_to": "MPI"
        },
        {
            "id": "q2", 
            "question": "Da li tim koristi OpenMP za paralelizaciju?",
            "options": [
                (0, "Ne koristi"),
                (1, "Poznaje osnove"),
                (2, "Koristi basic #pragma omp parallel"),
                (3, "Razumije scheduling, reduction"),
                (4, "Optimizuje sa tasks, SIMD")
            ],
            "maps_to": "OpenMP"
        }
    ],
    "hpc_tools": [
        {
            "id": "q3",
            "question": "Da li tim koristi job scheduler?",
            "options": [
                (0, "Ne"),
                (1, "Osnovne sbatch komande"),
                (2, "Kreira job skripte"),
                (3, "Koristi job arrays, dependencies"),
                (4, "Optimizuje scheduling politike")
            ],
            "maps_to": "slurm"
        }
    ]
}
