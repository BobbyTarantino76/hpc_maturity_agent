#!/usr/bin/env python3
"""
HPC Maturity Assessment Agent - GUI
====================================
Grafički korisnički interfejs za procjenu HPC zrelosti.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
from typing import Optional
import threading

from agent import HPCMaturityAgent, AssessmentInput, create_demo_data
from infrastructure_analyzer import (
    InfrastructureProfile, ComputeNode, Workload, ResourceMetrics
)
from team_evaluator import TeamProfile, TeamMember, SkillLevel
from software_analyzer import SoftwareProfile
from simulator import HPCClusterConfig


class ModernStyle:
    """Moderna tema za GUI."""
    BG_COLOR = "#1e1e2e"
    BG_SECONDARY = "#2d2d3f"
    BG_CARD = "#363649"
    TEXT_COLOR = "#cdd6f4"
    TEXT_SECONDARY = "#a6adc8"
    ACCENT = "#89b4fa"
    ACCENT_HOVER = "#b4befe"
    SUCCESS = "#a6e3a1"
    WARNING = "#f9e2af"
    ERROR = "#f38ba8"
    BORDER = "#45475a"


class HPCMaturityGUI:
    """Glavni GUI za HPC Maturity Assessment."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HPC Maturity Assessment Agent")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Agent
        self.agent = HPCMaturityAgent()
        self.assessment_output = None
        
        # Data storage
        self.nodes = []
        self.workloads = []
        self.team_members = []
        
        # Setup style
        self._setup_style()
        
        # Build UI
        self._build_ui()
        
    def _setup_style(self):
        """Setup ttk style."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure(".", 
                       background=ModernStyle.BG_COLOR,
                       foreground=ModernStyle.TEXT_COLOR,
                       fieldbackground=ModernStyle.BG_SECONDARY)
        
        style.configure("TFrame", background=ModernStyle.BG_COLOR)
        style.configure("Card.TFrame", background=ModernStyle.BG_CARD)
        
        style.configure("TLabel", 
                       background=ModernStyle.BG_COLOR,
                       foreground=ModernStyle.TEXT_COLOR,
                       font=("Segoe UI", 10))
        
        style.configure("Header.TLabel",
                       font=("Segoe UI", 14, "bold"),
                       foreground=ModernStyle.ACCENT)
        
        style.configure("Title.TLabel",
                       font=("Segoe UI", 20, "bold"),
                       foreground=ModernStyle.ACCENT)
        
        style.configure("TButton",
                       font=("Segoe UI", 10),
                       padding=(15, 8))
        
        style.configure("Accent.TButton",
                       font=("Segoe UI", 11, "bold"))
        
        style.configure("TNotebook", background=ModernStyle.BG_COLOR)
        style.configure("TNotebook.Tab", 
                       font=("Segoe UI", 10),
                       padding=(20, 10))
        
        style.configure("TEntry",
                       fieldbackground=ModernStyle.BG_SECONDARY,
                       foreground=ModernStyle.TEXT_COLOR)
        
        style.configure("TSpinbox",
                       fieldbackground=ModernStyle.BG_SECONDARY)
        
        style.configure("TCombobox",
                       fieldbackground=ModernStyle.BG_SECONDARY)
        
        style.configure("Treeview",
                       background=ModernStyle.BG_SECONDARY,
                       foreground=ModernStyle.TEXT_COLOR,
                       fieldbackground=ModernStyle.BG_SECONDARY,
                       font=("Segoe UI", 9))
        
        style.configure("Treeview.Heading",
                       font=("Segoe UI", 10, "bold"))
        
        self.root.configure(bg=ModernStyle.BG_COLOR)
    
    def _build_ui(self):
        """Build main UI."""
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        ttk.Label(header_frame, 
                 text="HPC Maturity Assessment Agent",
                 style="Title.TLabel").pack(side=tk.LEFT)
        
        # Quick actions
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="Ucitaj JSON", 
                  command=self._load_json).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Eksportuj", 
                  command=self._export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Demo", 
                  command=self._run_demo,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tabs
        self._build_org_tab()
        self._build_infrastructure_tab()
        self._build_team_tab()
        self._build_software_tab()
        self._build_cluster_tab()
        self._build_results_tab()
        
        # Bottom buttons
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.run_btn = ttk.Button(bottom_frame, 
                                  text="Pokreni Procjenu",
                                  command=self._run_assessment,
                                  style="Accent.TButton")
        self.run_btn.pack(side=tk.RIGHT, padx=5)
        
        self.progress_var = tk.StringVar(value="Spremno za procjenu")
        ttk.Label(bottom_frame, textvariable=self.progress_var).pack(side=tk.LEFT)
    
    def _build_org_tab(self):
        """Build organization tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Organizacija")
        
        # Organization name
        org_frame = ttk.LabelFrame(frame, text="Podaci o organizaciji", padding=15)
        org_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(org_frame, text="Naziv organizacije:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.org_name_var = tk.StringVar(value="Moja Organizacija")
        ttk.Entry(org_frame, textvariable=self.org_name_var, width=50).grid(row=0, column=1, pady=5, padx=10)
        
        # Info
        info_frame = ttk.LabelFrame(frame, text="O procjeni", padding=15)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        info_text = """
HPC Maturity Assessment Agent

Ovaj alat analizira spremnost vase organizacije za koristenje 
High-Performance Computing (HPC) infrastrukture.

Procjena ukljucuje:
   - Analizu postojece infrastrukture i workload-a
   - Evaluaciju vjestina i spremnosti tima
   - Procjenu optimizacije softvera
   - Simulaciju performansi na target HPC klasteru
   - Generisanje preporuka za migraciju

Kako koristiti:
   1. Unesite podatke o infrastrukturi (cvorovi, workload-i)
   2. Dodajte clanove tima i njihove vjestine
   3. Opisite softverski profil
   4. Konfigurisite target HPC klaster
   5. Pokrenite procjenu

Tip: Koristite "Demo" dugme za brzu demonstraciju 
   sa unaprijed definisanim podacima.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(fill=tk.BOTH, expand=True)
    
    def _build_infrastructure_tab(self):
        """Build infrastructure tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Infrastruktura")
        
        # Create paned window
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Nodes
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Nodes section
        nodes_frame = ttk.LabelFrame(left_frame, text="Računarski čvorovi", padding=10)
        nodes_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Node form
        form_frame = ttk.Frame(nodes_frame)
        form_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(form_frame, text="CPU jezgara:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.node_cores_var = tk.IntVar(value=32)
        ttk.Spinbox(form_frame, from_=1, to=256, textvariable=self.node_cores_var, width=10).grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(form_frame, text="CPU GHz:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.node_freq_var = tk.DoubleVar(value=2.5)
        ttk.Spinbox(form_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.node_freq_var, width=10).grid(row=0, column=3, pady=2, padx=5)
        
        ttk.Label(form_frame, text="Memorija (GB):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.node_mem_var = tk.IntVar(value=128)
        ttk.Spinbox(form_frame, from_=8, to=2048, increment=8, textvariable=self.node_mem_var, width=10).grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Label(form_frame, text="GPU:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.node_gpu_var = tk.IntVar(value=0)
        ttk.Spinbox(form_frame, from_=0, to=16, textvariable=self.node_gpu_var, width=10).grid(row=1, column=3, pady=2, padx=5)
        
        ttk.Button(form_frame, text="Dodaj cvor", command=self._add_node).grid(row=2, column=0, columnspan=4, pady=10)
        
        # Nodes list
        columns = ("id", "cores", "freq", "memory", "gpu")
        self.nodes_tree = ttk.Treeview(nodes_frame, columns=columns, show="headings", height=6)
        self.nodes_tree.heading("id", text="ID")
        self.nodes_tree.heading("cores", text="CPU")
        self.nodes_tree.heading("freq", text="GHz")
        self.nodes_tree.heading("memory", text="RAM (GB)")
        self.nodes_tree.heading("gpu", text="GPU")
        
        for col in columns:
            self.nodes_tree.column(col, width=70)
        
        self.nodes_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(nodes_frame, text="Obrisi odabrano", command=self._delete_node).pack(pady=5)
        
        # Right: Workloads and Config
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Workloads section
        workloads_frame = ttk.LabelFrame(right_frame, text="Workload-i", padding=10)
        workloads_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        wl_form = ttk.Frame(workloads_frame)
        wl_form.pack(fill=tk.X, pady=5)
        
        ttk.Label(wl_form, text="Naziv:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.wl_name_var = tk.StringVar(value="Workload")
        ttk.Entry(wl_form, textvariable=self.wl_name_var, width=15).grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(wl_form, text="Tip:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.wl_type_var = tk.StringVar(value="compute_intensive")
        type_combo = ttk.Combobox(wl_form, textvariable=self.wl_type_var, width=15,
                                  values=["compute_intensive", "memory_intensive", "io_intensive", "balanced", "gpu_accelerated"])
        type_combo.grid(row=0, column=3, pady=2, padx=5)
        
        ttk.Label(wl_form, text="CPU util (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.wl_cpu_var = tk.IntVar(value=70)
        ttk.Spinbox(wl_form, from_=0, to=100, textvariable=self.wl_cpu_var, width=10).grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Label(wl_form, text="Runtime (h):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.wl_runtime_var = tk.DoubleVar(value=8.0)
        ttk.Spinbox(wl_form, from_=0.1, to=1000, increment=0.5, textvariable=self.wl_runtime_var, width=10).grid(row=1, column=3, pady=2, padx=5)
        
        ttk.Label(wl_form, text="Parallel (0-1):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.wl_parallel_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(wl_form, from_=0, to=1, increment=0.05, textvariable=self.wl_parallel_var, width=10).grid(row=2, column=1, pady=2, padx=5)
        
        ttk.Label(wl_form, text="Data (GB):").grid(row=2, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.wl_data_var = tk.IntVar(value=100)
        ttk.Spinbox(wl_form, from_=1, to=100000, textvariable=self.wl_data_var, width=10).grid(row=2, column=3, pady=2, padx=5)
        
        ttk.Button(wl_form, text="Dodaj workload", command=self._add_workload).grid(row=3, column=0, columnspan=4, pady=10)
        
        # Workloads list
        wl_columns = ("name", "type", "cpu", "runtime", "parallel")
        self.workloads_tree = ttk.Treeview(workloads_frame, columns=wl_columns, show="headings", height=5)
        self.workloads_tree.heading("name", text="Naziv")
        self.workloads_tree.heading("type", text="Tip")
        self.workloads_tree.heading("cpu", text="CPU%")
        self.workloads_tree.heading("runtime", text="Runtime")
        self.workloads_tree.heading("parallel", text="Parallel")
        
        for col in wl_columns:
            self.workloads_tree.column(col, width=80)
        
        self.workloads_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(workloads_frame, text="Obrisi", command=self._delete_workload).pack(pady=5)
        
        # Config section
        config_frame = ttk.LabelFrame(right_frame, text="Konfiguracija", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        self.has_scheduler_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Job Scheduler", variable=self.has_scheduler_var).grid(row=0, column=0, sticky=tk.W)
        
        self.scheduler_type_var = tk.StringVar(value="none")
        ttk.Combobox(config_frame, textvariable=self.scheduler_type_var, width=12,
                    values=["none", "slurm", "pbs", "sge"]).grid(row=0, column=1, padx=5)
        
        self.has_shared_fs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Shared FS", variable=self.has_shared_fs_var).grid(row=1, column=0, sticky=tk.W)
        
        self.fs_type_var = tk.StringVar(value="local")
        ttk.Combobox(config_frame, textvariable=self.fs_type_var, width=12,
                    values=["local", "nfs", "lustre", "gpfs"]).grid(row=1, column=1, padx=5)
        
        ttk.Label(config_frame, text="Interconnect:").grid(row=2, column=0, sticky=tk.W)
        self.interconnect_var = tk.StringVar(value="ethernet")
        ttk.Combobox(config_frame, textvariable=self.interconnect_var, width=12,
                    values=["ethernet", "infiniband", "omnipath"]).grid(row=2, column=1, padx=5)
    
    def _build_team_tab(self):
        """Build team tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Tim")
        
        # Team name
        name_frame = ttk.Frame(frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Naziv tima:").pack(side=tk.LEFT)
        self.team_name_var = tk.StringVar(value="Development Team")
        ttk.Entry(name_frame, textvariable=self.team_name_var, width=30).pack(side=tk.LEFT, padx=10)
        
        # Paned window
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Member form
        left_frame = ttk.LabelFrame(paned, text="Dodaj člana tima", padding=10)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Ime:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.member_name_var = tk.StringVar(value="")
        ttk.Entry(left_frame, textvariable=self.member_name_var, width=20).grid(row=0, column=1, pady=3, padx=5)
        
        ttk.Label(left_frame, text="Uloga:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.member_role_var = tk.StringVar(value="Developer")
        ttk.Combobox(left_frame, textvariable=self.member_role_var, width=18,
                    values=["Developer", "Lead Developer", "Data Scientist", "DevOps Engineer", "Researcher"]).grid(row=1, column=1, pady=3, padx=5)
        
        ttk.Label(left_frame, text="God. iskustva:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.member_exp_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(left_frame, from_=0, to=40, textvariable=self.member_exp_var, width=10).grid(row=2, column=1, pady=3, padx=5, sticky=tk.W)
        
        ttk.Label(left_frame, text="HPC projekti:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.member_hpc_var = tk.IntVar(value=0)
        ttk.Spinbox(left_frame, from_=0, to=50, textvariable=self.member_hpc_var, width=10).grid(row=3, column=1, pady=3, padx=5, sticky=tk.W)
        
        # Skills
        ttk.Label(left_frame, text="Vještine (0-4):", style="Header.TLabel").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        skills_frame = ttk.Frame(left_frame)
        skills_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W)
        
        self.skill_vars = {}
        skills = ["MPI", "OpenMP", "CUDA", "slurm", "profilers", "version_control", "containers"]
        
        for i, skill in enumerate(skills):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(skills_frame, text=f"{skill}:").grid(row=row, column=col, sticky=tk.W, pady=2)
            self.skill_vars[skill] = tk.IntVar(value=0)
            ttk.Spinbox(skills_frame, from_=0, to=4, textvariable=self.skill_vars[skill], width=5).grid(row=row, column=col+1, pady=2, padx=(5, 15))
        
        ttk.Button(left_frame, text="Dodaj clana", command=self._add_member).grid(row=10, column=0, columnspan=2, pady=15)
        
        # Right: Members list
        right_frame = ttk.LabelFrame(paned, text="Članovi tima", padding=10)
        paned.add(right_frame, weight=1)
        
        columns = ("name", "role", "exp", "hpc", "skills")
        self.members_tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=10)
        self.members_tree.heading("name", text="Ime")
        self.members_tree.heading("role", text="Uloga")
        self.members_tree.heading("exp", text="Iskustvo")
        self.members_tree.heading("hpc", text="HPC")
        self.members_tree.heading("skills", text="Top vještine")
        
        self.members_tree.column("name", width=100)
        self.members_tree.column("role", width=100)
        self.members_tree.column("exp", width=60)
        self.members_tree.column("hpc", width=50)
        self.members_tree.column("skills", width=150)
        
        self.members_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(right_frame, text="Obrisi odabrano", command=self._delete_member).pack(pady=5)
        
        # Team config
        config_frame = ttk.LabelFrame(frame, text="Timski procesi", padding=10)
        config_frame.pack(fill=tk.X, pady=10)
        
        self.has_training_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="HPC Training Program", variable=self.has_training_var).pack(side=tk.LEFT, padx=10)
        
        self.has_docs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Dokumentacija", variable=self.has_docs_var).pack(side=tk.LEFT, padx=10)
        
        self.has_review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Code Review", variable=self.has_review_var).pack(side=tk.LEFT, padx=10)
    
    def _build_software_tab(self):
        """Build software tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Softver")
        
        # Left column
        left_frame = ttk.LabelFrame(frame, text="Softverski profil", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(left_frame, text="Naziv aplikacije:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sw_name_var = tk.StringVar(value="MyApp")
        ttk.Entry(left_frame, textvariable=self.sw_name_var, width=25).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(left_frame, text="Programski jezik:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sw_lang_var = tk.StringVar(value="python")
        ttk.Combobox(left_frame, textvariable=self.sw_lang_var, width=22,
                    values=["python", "c", "cpp", "fortran", "java"]).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(left_frame, text="Linije koda (procjena):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.sw_loc_var = tk.IntVar(value=5000)
        ttk.Spinbox(left_frame, from_=100, to=10000000, textvariable=self.sw_loc_var, width=15).grid(row=2, column=1, pady=5, padx=5, sticky=tk.W)
        
        ttk.Label(left_frame, text="Max testirani cores:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.sw_cores_var = tk.IntVar(value=4)
        ttk.Spinbox(left_frame, from_=1, to=10000, textvariable=self.sw_cores_var, width=15).grid(row=3, column=1, pady=5, padx=5, sticky=tk.W)
        
        # Technologies
        ttk.Label(left_frame, text="Tehnologije:", style="Header.TLabel").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        self.sw_mpi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Koristi MPI", variable=self.sw_mpi_var).grid(row=5, column=0, sticky=tk.W)
        
        self.sw_openmp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Koristi OpenMP", variable=self.sw_openmp_var).grid(row=5, column=1, sticky=tk.W)
        
        self.sw_cuda_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Koristi CUDA", variable=self.sw_cuda_var).grid(row=6, column=0, sticky=tk.W)
        
        self.sw_vec_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Vektorizacija", variable=self.sw_vec_var).grid(row=6, column=1, sticky=tk.W)
        
        self.sw_profiling_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Profiling podaci", variable=self.sw_profiling_var).grid(row=7, column=0, sticky=tk.W)
        
        self.sw_scalability_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Testirana skalabilnost", variable=self.sw_scalability_var).grid(row=7, column=1, sticky=tk.W)
        
        # Right column: Code sample
        right_frame = ttk.LabelFrame(frame, text="Uzorak koda (opciono)", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.code_text = scrolledtext.ScrolledText(right_frame, width=50, height=20,
                                                   bg=ModernStyle.BG_SECONDARY,
                                                   fg=ModernStyle.TEXT_COLOR,
                                                   insertbackground=ModernStyle.TEXT_COLOR,
                                                   font=("Consolas", 10))
        self.code_text.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="Unesite uzorak koda za analizu anti-pattern-a",
                 foreground=ModernStyle.TEXT_SECONDARY).pack(pady=5)
    
    def _build_cluster_tab(self):
        """Build target cluster tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Target Klaster")
        
        config_frame = ttk.LabelFrame(frame, text="Konfiguracija target HPC klastera", padding=15)
        config_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(config_frame, text="Naziv klastera:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.cluster_name_var = tk.StringVar(value="Target HPC Cluster")
        ttk.Entry(config_frame, textvariable=self.cluster_name_var, width=30).grid(row=0, column=1, pady=8, padx=10)
        
        ttk.Label(config_frame, text="Broj čvorova:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.cluster_nodes_var = tk.IntVar(value=20)
        ttk.Spinbox(config_frame, from_=1, to=1000, textvariable=self.cluster_nodes_var, width=15).grid(row=1, column=1, pady=8, padx=10, sticky=tk.W)
        
        ttk.Label(config_frame, text="Jezgara po čvoru:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.cluster_cores_var = tk.IntVar(value=64)
        ttk.Spinbox(config_frame, from_=1, to=256, textvariable=self.cluster_cores_var, width=15).grid(row=2, column=1, pady=8, padx=10, sticky=tk.W)
        
        ttk.Label(config_frame, text="Memorija po čvoru (GB):").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.cluster_mem_var = tk.IntVar(value=256)
        ttk.Spinbox(config_frame, from_=8, to=4096, textvariable=self.cluster_mem_var, width=15).grid(row=3, column=1, pady=8, padx=10, sticky=tk.W)
        
        ttk.Label(config_frame, text="GPU po čvoru:").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.cluster_gpu_var = tk.IntVar(value=4)
        ttk.Spinbox(config_frame, from_=0, to=16, textvariable=self.cluster_gpu_var, width=15).grid(row=4, column=1, pady=8, padx=10, sticky=tk.W)
        
        ttk.Label(config_frame, text="Paralelna efikasnost (0-1):").grid(row=5, column=0, sticky=tk.W, pady=8)
        self.cluster_eff_var = tk.DoubleVar(value=0.85)
        ttk.Spinbox(config_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.cluster_eff_var, width=15).grid(row=5, column=1, pady=8, padx=10, sticky=tk.W)
        
        # Summary
        summary_frame = ttk.LabelFrame(frame, text="Sažetak kapaciteta", padding=15)
        summary_frame.pack(fill=tk.X, pady=10)
        
        self.cluster_summary_var = tk.StringVar()
        self._update_cluster_summary()
        
        ttk.Label(summary_frame, textvariable=self.cluster_summary_var,
                 font=("Segoe UI", 11)).pack(pady=10)
        
        ttk.Button(summary_frame, text="Osvjezi sazetak", 
                  command=self._update_cluster_summary).pack(pady=5)
    
    def _build_results_tab(self):
        """Build results tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Rezultati")
        
        # Score display
        score_frame = ttk.Frame(frame)
        score_frame.pack(fill=tk.X, pady=10)
        
        self.score_label = ttk.Label(score_frame, text="Maturity Score: --/100",
                                     style="Title.TLabel")
        self.score_label.pack()
        
        self.level_label = ttk.Label(score_frame, text="Nivo: --",
                                     font=("Segoe UI", 14))
        self.level_label.pack()
        
        # Component scores
        components_frame = ttk.LabelFrame(frame, text="Komponente", padding=10)
        components_frame.pack(fill=tk.X, pady=10)
        
        self.infra_score_var = tk.StringVar(value="Infrastruktura: --")
        self.team_score_var = tk.StringVar(value="Tim: --")
        self.software_score_var = tk.StringVar(value="Softver: --")
        
        ttk.Label(components_frame, textvariable=self.infra_score_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(components_frame, textvariable=self.team_score_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(components_frame, textvariable=self.software_score_var).pack(side=tk.LEFT, padx=20)
        
        # Results text
        self.results_text = scrolledtext.ScrolledText(frame, width=100, height=25,
                                                      bg=ModernStyle.BG_SECONDARY,
                                                      fg=ModernStyle.TEXT_COLOR,
                                                      font=("Consolas", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Maturity Map", 
                  command=self._show_maturity_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Detaljan izvjestaj", 
                  command=self._show_detailed_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Sacuvaj izvjestaj", 
                  command=self._save_report).pack(side=tk.LEFT, padx=5)
    
    # === Data Management Methods ===
    
    def _add_node(self):
        """Add a compute node."""
        node_id = f"node{len(self.nodes)+1:03d}"
        node = ComputeNode(
            node_id=node_id,
            cpu_cores=self.node_cores_var.get(),
            cpu_frequency_ghz=self.node_freq_var.get(),
            memory_gb=self.node_mem_var.get(),
            gpu_count=self.node_gpu_var.get()
        )
        self.nodes.append(node)
        
        self.nodes_tree.insert("", tk.END, values=(
            node_id,
            node.cpu_cores,
            f"{node.cpu_frequency_ghz:.1f}",
            node.memory_gb,
            node.gpu_count
        ))
    
    def _delete_node(self):
        """Delete selected node."""
        selected = self.nodes_tree.selection()
        if selected:
            idx = self.nodes_tree.index(selected[0])
            self.nodes_tree.delete(selected)
            if idx < len(self.nodes):
                self.nodes.pop(idx)
    
    def _add_workload(self):
        """Add a workload."""
        wl_id = f"wl{len(self.workloads)+1:03d}"
        workload = Workload(
            workload_id=wl_id,
            name=self.wl_name_var.get(),
            workload_type=self.wl_type_var.get(),
            avg_cpu_utilization=self.wl_cpu_var.get(),
            avg_memory_utilization=50,
            avg_io_operations_per_sec=100,
            avg_runtime_hours=self.wl_runtime_var.get(),
            parallelizable_fraction=self.wl_parallel_var.get(),
            data_size_gb=self.wl_data_var.get(),
            frequency_per_month=10
        )
        self.workloads.append(workload)
        
        self.workloads_tree.insert("", tk.END, values=(
            workload.name,
            workload.workload_type[:10],
            workload.avg_cpu_utilization,
            f"{workload.avg_runtime_hours:.1f}h",
            f"{workload.parallelizable_fraction:.2f}"
        ))
    
    def _delete_workload(self):
        """Delete selected workload."""
        selected = self.workloads_tree.selection()
        if selected:
            idx = self.workloads_tree.index(selected[0])
            self.workloads_tree.delete(selected)
            if idx < len(self.workloads):
                self.workloads.pop(idx)
    
    def _add_member(self):
        """Add a team member."""
        if not self.member_name_var.get():
            messagebox.showwarning("Upozorenje", "Unesite ime člana tima.")
            return
        
        member_id = f"m{len(self.team_members)+1:03d}"
        
        skills = {}
        skill_levels = [SkillLevel.NONE, SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 
                       SkillLevel.ADVANCED, SkillLevel.EXPERT]
        
        for skill, var in self.skill_vars.items():
            level = var.get()
            skills[skill] = skill_levels[min(level, 4)]
        
        member = TeamMember(
            member_id=member_id,
            name=self.member_name_var.get(),
            role=self.member_role_var.get(),
            years_experience=self.member_exp_var.get(),
            skills=skills,
            hpc_projects_completed=self.member_hpc_var.get()
        )
        self.team_members.append(member)
        
        # Get top skills
        top_skills = [s for s, l in skills.items() if l.value >= 2][:3]
        
        self.members_tree.insert("", tk.END, values=(
            member.name,
            member.role,
            f"{member.years_experience:.0f}g",
            member.hpc_projects_completed,
            ", ".join(top_skills) if top_skills else "-"
        ))
        
        # Clear form
        self.member_name_var.set("")
        for var in self.skill_vars.values():
            var.set(0)
    
    def _delete_member(self):
        """Delete selected member."""
        selected = self.members_tree.selection()
        if selected:
            idx = self.members_tree.index(selected[0])
            self.members_tree.delete(selected)
            if idx < len(self.team_members):
                self.team_members.pop(idx)
    
    def _update_cluster_summary(self):
        """Update cluster summary text."""
        nodes = self.cluster_nodes_var.get()
        cores = self.cluster_cores_var.get()
        mem = self.cluster_mem_var.get()
        gpus = self.cluster_gpu_var.get()
        
        total_cores = nodes * cores
        total_mem = nodes * mem
        total_gpus = nodes * gpus
        
        summary = f"""
Ukupni kapacitet:
   - Cvorova: {nodes}
   - CPU jezgara: {total_cores:,}
   - Memorije: {total_mem:,} GB ({total_mem/1024:.1f} TB)
   - GPU: {total_gpus}
   - Procijenjeni TFLOPS: ~{total_cores * 0.05:.1f} (CPU) + ~{total_gpus * 10:.1f} (GPU)
        """
        self.cluster_summary_var.set(summary)
    
    # === Assessment Methods ===
    
    def _collect_input_data(self) -> AssessmentInput:
        """Collect all input data for assessment."""
        # Infrastructure
        metrics = ResourceMetrics(
            avg_cpu_utilization=50,
            peak_cpu_utilization=85,
            avg_memory_utilization=45,
            peak_memory_utilization=80,
            idle_time_percentage=20
        )
        
        infrastructure = None
        if self.nodes:
            infrastructure = InfrastructureProfile(
                organization_name=self.org_name_var.get(),
                nodes=self.nodes,
                workloads=self.workloads if self.workloads else [],
                metrics=metrics,
                has_job_scheduler=self.has_scheduler_var.get(),
                scheduler_type=self.scheduler_type_var.get(),
                has_shared_filesystem=self.has_shared_fs_var.get(),
                filesystem_type=self.fs_type_var.get(),
                interconnect_type=self.interconnect_var.get()
            )
        
        # Team
        team = None
        if self.team_members:
            team = TeamProfile(
                team_name=self.team_name_var.get(),
                members=self.team_members,
                has_hpc_training_program=self.has_training_var.get(),
                has_documentation=self.has_docs_var.get(),
                has_code_review_process=self.has_review_var.get(),
                collaboration_tools=["Git", "Slack"]
            )
        
        # Software
        code = self.code_text.get("1.0", tk.END).strip()
        software = SoftwareProfile(
            name=self.sw_name_var.get(),
            language=self.sw_lang_var.get(),
            source_code=code if code else None,
            estimated_loc=self.sw_loc_var.get(),
            uses_mpi=self.sw_mpi_var.get(),
            uses_openmp=self.sw_openmp_var.get(),
            uses_cuda=self.sw_cuda_var.get(),
            uses_vectorization=self.sw_vec_var.get(),
            has_profiling_data=self.sw_profiling_var.get(),
            scalability_tested=self.sw_scalability_var.get(),
            max_tested_cores=self.sw_cores_var.get()
        )
        
        # Target cluster
        target_cluster = HPCClusterConfig(
            name=self.cluster_name_var.get(),
            total_nodes=self.cluster_nodes_var.get(),
            cores_per_node=self.cluster_cores_var.get(),
            memory_per_node_gb=self.cluster_mem_var.get(),
            gpus_per_node=self.cluster_gpu_var.get(),
            parallel_efficiency=self.cluster_eff_var.get()
        )
        
        return AssessmentInput(
            organization_name=self.org_name_var.get(),
            infrastructure=infrastructure,
            team=team,
            software=software,
            target_cluster=target_cluster
        )
    
    def _run_assessment(self):
        """Run the assessment."""
        self.progress_var.set("Pokrecem procjenu...")
        self.run_btn.config(state=tk.DISABLED)
        
        def run():
            try:
                input_data = self._collect_input_data()
                self.assessment_output = self.agent.run_assessment(input_data)
                self.root.after(0, self._update_results)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Greška", str(e)))
            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.progress_var.set("Procjena zavrsena"))
        
        thread = threading.Thread(target=run)
        thread.start()
    
    def _run_demo(self):
        """Run demo assessment."""
        self.progress_var.set("Pokrecem demo procjenu...")
        self.run_btn.config(state=tk.DISABLED)
        
        def run():
            try:
                demo_input = create_demo_data()
                self.assessment_output = self.agent.run_assessment(demo_input)
                self.root.after(0, self._update_results)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Greška", str(e)))
            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.progress_var.set("Demo procjena zavrsena"))
        
        thread = threading.Thread(target=run)
        thread.start()
    
    def _update_results(self):
        """Update results display."""
        if not self.assessment_output:
            return
        
        out = self.assessment_output
        
        # Update scores
        self.score_label.config(text=f"Maturity Score: {out.maturity_score:.1f}/100")
        self.level_label.config(text=f"Nivo {out.maturity_level}: {out.maturity_level_name}")
        
        self.infra_score_var.set(f"Infrastruktura: {out.infrastructure_score:.1f}")
        self.team_score_var.set(f"Tim: {out.team_score:.1f}")
        self.software_score_var.set(f"Softver: {out.software_score:.1f}")
        
        # Update text
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, out.summary)
        
        # Switch to results tab
        self.notebook.select(5)
    
    def _show_maturity_map(self):
        """Show maturity map."""
        if not self.agent.assessment_complete:
            messagebox.showinfo("Info", "Prvo pokrenite procjenu.")
            return
        
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, self.agent.get_maturity_map())
    
    def _show_detailed_report(self):
        """Show detailed report."""
        if not self.agent.assessment_complete:
            messagebox.showinfo("Info", "Prvo pokrenite procjenu.")
            return
        
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, self.agent.get_detailed_report())
    
    def _save_report(self):
        """Save report to file."""
        if not self.agent.assessment_complete:
            messagebox.showinfo("Info", "Prvo pokrenite procjenu.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfilename="hpc_assessment_report.txt"
        )
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.agent.get_maturity_map())
                f.write("\n\n")
                f.write(self.assessment_output.summary)
                f.write("\n\n")
                f.write(self.agent.get_detailed_report())
            
            messagebox.showinfo("Uspjeh", f"Izvještaj sačuvan: {filepath}")
    
    def _export_results(self):
        """Export results to JSON."""
        if not self.agent.assessment_complete:
            messagebox.showinfo("Info", "Prvo pokrenite procjenu.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfilename="hpc_assessment_results.json"
        )
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.agent.export_results("json"))
            
            messagebox.showinfo("Uspjeh", f"Rezultati eksportovani: {filepath}")
    
    def _load_json(self):
        """Load data from JSON file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                messagebox.showinfo("Info", "JSON učitan. Funkcionalnost za parsiranje u razvoju.")
            except Exception as e:
                messagebox.showerror("Greška", f"Greška pri učitavanju: {e}")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = HPCMaturityGUI()
    app.run()


if __name__ == "__main__":
    main()
