#!/usr/bin/env python3
"""
HPC Maturity Assessment Agent - Interaktivni CLI
=================================================
Interaktivni interfejs za prikupljanje podataka i
procjenu HPC zrelosti organizacije.
"""

import json
import os
import sys
from typing import Optional

from agent import (
    HPCMaturityAgent, AssessmentInput, create_demo_data
)
from infrastructure_analyzer import (
    InfrastructureProfile, ComputeNode, Workload, ResourceMetrics
)
from team_evaluator import (
    TeamProfile, TeamMember, SkillLevel
)
from software_analyzer import SoftwareProfile
from simulator import HPCClusterConfig


class InteractiveCLI:
    """Interaktivni CLI za HPC Maturity Assessment."""
    
    def __init__(self):
        self.agent = HPCMaturityAgent()
        self.input_data = None
    
    def clear_screen(self):
        """Očisti ekran."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self, title: str):
        """Štampaj header."""
        print("\n" + "═" * 60)
        print(f"  {title}")
        print("═" * 60 + "\n")
    
    def print_menu(self, options: list):
        """Štampaj meni opcija."""
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        print()
    
    def get_input(self, prompt: str, default: str = "") -> str:
        """Dohvati input od korisnika."""
        if default:
            result = input(f"  {prompt} [{default}]: ").strip()
            return result if result else default
        return input(f"  {prompt}: ").strip()
    
    def get_number(self, prompt: str, default: float = 0) -> float:
        """Dohvati numerički input."""
        while True:
            try:
                result = self.get_input(prompt, str(default))
                return float(result)
            except ValueError:
                print("   Molimo unesite broj.")
    
    def get_int(self, prompt: str, default: int = 0) -> int:
        """Dohvati integer input."""
        return int(self.get_number(prompt, default))
    
    def get_bool(self, prompt: str, default: bool = False) -> bool:
        """Dohvati boolean input (samo 1 ili 0)."""
        default_str = "1" if default else "0"
        while True:
            result = self.get_input(f"{prompt} (1=da, 0=ne)", default_str).strip()
            if result in ["1", "0"]:
                return result == "1"
            print("  Unesite 1 ili 0.")
    
    def run(self):
        """Pokreni interaktivni CLI."""
        self.clear_screen()
        self.print_header("HPC MATURITY ASSESSMENT AGENT")
        
        print("  Dobrodošli u AI Agent za procjenu zrelosti za HPC!")
        print("  Ovaj agent će analizirati vašu infrastrukturu, tim i softver")
        print("  te generisati preporuke za migraciju na HPC okruženje.\n")
        
        self.print_menu([
            "Pokreni demo procjenu (sa unaprijed definisanim podacima)",
            "Interaktivno unesi podatke",
            "Učitaj podatke iz JSON fajla",
            "Izlaz"
        ])
        
        choice = self.get_input("Odaberite opciju", "1")
        
        if choice == "1":
            self.run_demo()
        elif choice == "2":
            self.run_interactive()
        elif choice == "3":
            self.run_from_file()
        else:
            print("\n  Doviđenja!\n")
            sys.exit(0)
    
    def run_demo(self):
        """Pokreni demo procjenu."""
        self.print_header("DEMO PROCJENA")
        
        print("  Učitavam demo podatke...")
        self.input_data = create_demo_data()
        
        self.execute_assessment()
    
    def run_interactive(self):
        """Pokreni interaktivni unos."""
        self.print_header("INTERAKTIVNI UNOS PODATAKA")
        
        org_name = self.get_input("Naziv organizacije", "Moja Organizacija")
        
        # Prikupi podatke o infrastrukturi
        print("\nINFRASTRUKTURA\n")
        
        infrastructure = self.collect_infrastructure_data()
        
        # Prikupi podatke o timu
        print("\nTIM\n")
        
        team = self.collect_team_data()
        
        # Prikupi podatke o softveru
        print("\nSOFTVER\n")
        
        software = self.collect_software_data()
        
        # Target klaster
        print("\n TARGET HPC KLASTER\n")
        
        target_cluster = self.collect_cluster_config()
        
        self.input_data = AssessmentInput(
            organization_name=org_name,
            infrastructure=infrastructure,
            team=team,
            software=software,
            target_cluster=target_cluster
        )
        
        self.execute_assessment()
    
    def collect_infrastructure_data(self) -> InfrastructureProfile:
        """Prikupi podatke o infrastrukturi."""
        
        # Čvorovi
        node_count = self.get_int("Broj računarskih čvorova", 4)
        nodes = []
        
        print(f"\n  Unos podataka za {node_count} čvorova:\n")
        
        for i in range(node_count):
            print(f"  --- Čvor {i+1} ---")
            node = ComputeNode(
                node_id=f"node{i+1:03d}",
                cpu_cores=self.get_int("    CPU jezgara", 32),
                cpu_frequency_ghz=self.get_number("    CPU frekvencija (GHz)", 2.5),
                memory_gb=self.get_number("    Memorija (GB)", 128),
                gpu_count=self.get_int("    Broj GPU-a", 0),
                gpu_memory_gb=self.get_number("    GPU memorija (GB)", 0) if self.get_int("    Broj GPU-a", 0) > 0 else 0
            )
            nodes.append(node)
        
        # Workloads
        print("\n  Unos workload-a:\n")
        workload_count = self.get_int("Broj workload-a za analizu", 2)
        workloads = []
        
        workload_types = ["compute_intensive", "memory_intensive", 
                         "io_intensive", "balanced", "gpu_accelerated"]
        
        for i in range(workload_count):
            print(f"\n  --- Workload {i+1} ---")
            name = self.get_input("    Naziv", f"Workload_{i+1}")
            
            print("    Tipovi: 1=compute, 2=memory, 3=io, 4=balanced, 5=gpu")
            type_idx = self.get_int("    Tip (1-5)", 1) - 1
            wl_type = workload_types[max(0, min(4, type_idx))]
            
            workload = Workload(
                workload_id=f"wl{i+1:03d}",
                name=name,
                workload_type=wl_type,
                avg_cpu_utilization=self.get_number("    Prosječna CPU utilizacija (%)", 60),
                avg_memory_utilization=self.get_number("    Prosječna Memory utilizacija (%)", 50),
                avg_io_operations_per_sec=self.get_number("    Prosječni IOPS", 100),
                avg_runtime_hours=self.get_number("    Prosječno vrijeme izvršenja (sati)", 8),
                parallelizable_fraction=self.get_number("    Paralelizabilni dio (0-1)", 0.7),
                data_size_gb=self.get_number("    Veličina podataka (GB)", 100),
                frequency_per_month=self.get_int("    Učestalost mjesečno", 10)
            )
            workloads.append(workload)
        
        # Metrike
        print("\n  Metrike iskorištenosti:\n")
        metrics = ResourceMetrics(
            avg_cpu_utilization=self.get_number("  Prosječna CPU utilizacija (%)", 50),
            peak_cpu_utilization=self.get_number("  Vršna CPU utilizacija (%)", 85),
            avg_memory_utilization=self.get_number("  Prosječna Memory utilizacija (%)", 45),
            peak_memory_utilization=self.get_number("  Vršna Memory utilizacija (%)", 80),
            idle_time_percentage=self.get_number("  Procenat neaktivnosti (%)", 20)
        )
        
        # Konfiguracija
        print("\n  Konfiguracija infrastrukture:\n")
        
        has_scheduler = self.get_bool("  Da li imate job scheduler", False)
        scheduler_type = "none"
        if has_scheduler:
            scheduler_type = self.get_input("    Tip scheduler-a (slurm/pbs/sge)", "slurm")
        
        has_shared_fs = self.get_bool("  Da li imate dijeljeni filesystem", False)
        fs_type = "local"
        if has_shared_fs:
            fs_type = self.get_input("    Tip filesystem-a (lustre/gpfs/nfs)", "nfs")
        
        interconnect = self.get_input("  Interconnect tip (ethernet/infiniband)", "ethernet")
        
        return InfrastructureProfile(
            organization_name="",
            nodes=nodes,
            workloads=workloads,
            metrics=metrics,
            has_job_scheduler=has_scheduler,
            scheduler_type=scheduler_type,
            has_shared_filesystem=has_shared_fs,
            filesystem_type=fs_type,
            interconnect_type=interconnect
        )
    
    def collect_team_data(self) -> TeamProfile:
        """Prikupi podatke o timu."""
        
        team_name = self.get_input("Naziv tima", "Development Team")
        member_count = self.get_int("Broj članova tima", 3)
        
        members = []
        skill_options = {
            0: SkillLevel.NONE,
            1: SkillLevel.BEGINNER,
            2: SkillLevel.INTERMEDIATE,
            3: SkillLevel.ADVANCED,
            4: SkillLevel.EXPERT
        }
        
        for i in range(member_count):
            print(f"\n  --- Član tima {i+1} ---")
            name = self.get_input("    Ime", f"Član {i+1}")
            role = self.get_input("    Uloga", "Developer")
            years = self.get_number("    Godine iskustva", 3)
            hpc_projects = self.get_int("    Broj završenih HPC projekata", 0)
            
            print("    Vještine (0=nema, 1=početnik, 2=srednji, 3=napredan, 4=ekspert):")
            skills = {}
            
            for skill in ["MPI", "OpenMP", "CUDA", "slurm", "profilers"]:
                level = self.get_int(f"      {skill}", 0)
                skills[skill] = skill_options.get(level, SkillLevel.NONE)
            
            member = TeamMember(
                member_id=f"m{i+1:03d}",
                name=name,
                role=role,
                years_experience=years,
                skills=skills,
                hpc_projects_completed=hpc_projects
            )
            members.append(member)
        
        print("\n  Timski procesi:\n")
        has_training = self.get_bool("  Da li imate HPC training program", False)
        has_docs = self.get_bool("  Da li imate dokumentaciju", False)
        has_review = self.get_bool("  Da li imate code review proces", False)
        
        tools = self.get_input("  Kolaboracioni alati (comma-separated)", "Git,Slack")
        
        return TeamProfile(
            team_name=team_name,
            members=members,
            has_hpc_training_program=has_training,
            has_documentation=has_docs,
            has_code_review_process=has_review,
            collaboration_tools=[t.strip() for t in tools.split(",")]
        )
    
    def collect_software_data(self) -> SoftwareProfile:
        """Prikupi podatke o softveru."""
        
        name = self.get_input("Naziv aplikacije", "MyApp")
        language = self.get_input("Programski jezik (python/c/cpp/fortran)", "python")
        loc = self.get_int("Procijenjeni broj linija koda", 1000)
        
        print("\n  Korištene tehnologije:\n")
        uses_mpi = self.get_bool("  Koristi MPI", False)
        uses_openmp = self.get_bool("  Koristi OpenMP", False)
        uses_cuda = self.get_bool("  Koristi CUDA", False)
        uses_vec = self.get_bool("  Koristi vektorizaciju (NumPy/SIMD)", False)
        
        has_profiling = self.get_bool("  Postoje podaci o profilisanju", False)
        tested = self.get_bool("  Testirana skalabilnost", False)
        max_cores = self.get_int("  Maksimalni testirani broj jezgara", 1)
        
        efficiency = None
        if tested and max_cores > 1:
            efficiency = self.get_number("  Izmjerena paralelna efikasnost (0-1)", 0.7)
        
        # Opciono: unos koda
        code = None
        if self.get_bool("\n  Da li želite unijeti uzorak koda za analizu", False):
            print("  Unesite kod (završite sa praznim redom):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            code = "\n".join(lines)
        
        return SoftwareProfile(
            name=name,
            language=language,
            source_code=code,
            estimated_loc=loc,
            uses_mpi=uses_mpi,
            uses_openmp=uses_openmp,
            uses_cuda=uses_cuda,
            uses_vectorization=uses_vec,
            has_profiling_data=has_profiling,
            measured_parallel_efficiency=efficiency,
            scalability_tested=tested,
            max_tested_cores=max_cores
        )
    
    def collect_cluster_config(self) -> HPCClusterConfig:
        """Prikupi konfiguraciju target klastera."""
        
        if not self.get_bool("Da li želite simulirati na target HPC klasteru", True):
            return None
        
        return HPCClusterConfig(
            name=self.get_input("  Naziv klastera", "Target HPC"),
            total_nodes=self.get_int("  Broj čvorova", 20),
            cores_per_node=self.get_int("  Jezgara po čvoru", 64),
            memory_per_node_gb=self.get_number("  Memorija po čvoru (GB)", 256),
            gpus_per_node=self.get_int("  GPU po čvoru", 2),
            parallel_efficiency=self.get_number("  Očekivana paralelna efikasnost (0-1)", 0.85)
        )
    
    def run_from_file(self):
        """Učitaj podatke iz JSON fajla."""
        self.print_header("UČITAVANJE IZ FAJLA")
        
        filepath = self.get_input("Putanja do JSON fajla", "input.json")
        
        if not os.path.exists(filepath):
            print(f"   Fajl '{filepath}' ne postoji.")
            print("  Pokrećem demo procjenu...\n")
            self.run_demo()
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Parsiraj JSON u objekte
            # (pojednostavljena verzija, treba proširiti za punu funkcionalnost)
            print("  Fajl učitan.")
            print("   Za sada, koristim demo podatke...")
            
            self.input_data = create_demo_data()
            self.execute_assessment()
            
        except Exception as e:
            print(f"   Greška pri čitanju fajla: {e}")
            print("  Pokrećem demo procjenu...\n")
            self.run_demo()
    
    def execute_assessment(self):
        """Izvrši procjenu i prikaži rezultate."""
        self.print_header("POKRETANJE PROCJENE")
        
        # Pokreni agenta
        output = self.agent.run_assessment(self.input_data)
        
        # Prikaži maturity mapu
        print(self.agent.get_maturity_map())
        
        # Prikaži executive summary
        print(output.summary)
        
        # Meni za dalje akcije
        while True:
            self.print_menu([
                "Prikaži detaljan izvještaj",
                "Prikaži maturity mapu",
                "Eksportuj rezultate (JSON)",
                "Sačuvaj izvještaj u fajl",
                "Izlaz"
            ])
            
            choice = self.get_input("Odaberite opciju", "5")
            
            if choice == "1":
                print(self.agent.get_detailed_report())
            elif choice == "2":
                print(self.agent.get_maturity_map())
            elif choice == "3":
                print(self.agent.export_results("json"))
            elif choice == "4":
                filename = self.get_input("Naziv fajla", "hpc_assessment_report.txt")
                with open(filename, 'w') as f:
                    f.write(self.agent.get_maturity_map())
                    f.write("\n\n")
                    f.write(output.summary)
                    f.write("\n\n")
                    f.write(self.agent.get_detailed_report())
                print(f"  Izvještaj sačuvan u '{filename}'")
            else:
                print("\n  Hvala na korištenju HPC Maturity Assessment Agent-a!")
                print("  Doviđenja!\n")
                break


def main():
    """Main entry point."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
