"""
Modul za analizu dostupnosti podataka i nivoa interoperabilnosti.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DataAccessLevel(Enum):
    """Nivoi pristupa podacima."""
    NONE = 0
    LIMITED = 1
    PARTIAL = 2
    FULL = 3
    REAL_TIME = 4


class DataFormat(Enum):
    """Formati podataka."""
    PROPRIETARY = "proprietary"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    HDF5 = "hdf5"
    NETCDF = "netcdf"
    PARQUET = "parquet"
    BINARY = "binary"


class InteroperabilityLevel(Enum):
    """Nivoi interoperabilnosti."""
    ISOLATED = 1        # Nema integracije
    POINT_TO_POINT = 2  # Ad-hoc integracije
    STANDARDIZED = 3    # Koristi standarde
    INTEGRATED = 4      # Potpuna integracija
    SEAMLESS = 5        # Transparentna interoperabilnost


@dataclass
class DataSource:
    """Izvor podataka."""
    name: str
    format: DataFormat
    access_level: DataAccessLevel
    size_gb: float = 0.0
    update_frequency: str = "static"  # static, daily, hourly, real-time
    has_api: bool = False
    has_documentation: bool = False
    has_metadata: bool = False
    is_validated: bool = False


@dataclass
class SystemInterface:
    """Interfejs sistema."""
    name: str
    protocol: str  # REST, SOAP, MPI-IO, POSIX, custom
    is_standard: bool = False
    has_authentication: bool = False
    has_versioning: bool = False
    latency_ms: float = 0.0
    throughput_gbps: float = 0.0


@dataclass 
class DataInteroperabilityProfile:
    """Profil dostupnosti podataka i interoperabilnosti."""
    organization_name: str
    data_sources: List[DataSource] = field(default_factory=list)
    system_interfaces: List[SystemInterface] = field(default_factory=list)
    
    # Data governance
    has_data_catalog: bool = False
    has_data_lineage: bool = False
    has_data_quality_checks: bool = False
    has_backup_strategy: bool = False
    
    # Standards
    uses_standard_formats: bool = False
    uses_standard_protocols: bool = False
    has_data_standards_policy: bool = False
    
    # Integration
    has_etl_pipeline: bool = False
    has_data_warehouse: bool = False
    has_api_gateway: bool = False


class DataInteroperabilityAnalyzer:
    """Analizator dostupnosti podataka i interoperabilnosti."""
    
    def __init__(self, profile: DataInteroperabilityProfile):
        self.profile = profile
        self.results = {}
    
    def analyze(self) -> Dict:
        """Izvrši kompletnu analizu."""
        self.results = {
            "data_availability": self._analyze_data_availability(),
            "data_quality": self._analyze_data_quality(),
            "interoperability": self._analyze_interoperability(),
            "integration_readiness": self._analyze_integration_readiness(),
            "recommendations": self._generate_recommendations(),
            "data_interop_score": 0.0
        }
        
        # Izracunaj ukupni score
        self.results["data_interop_score"] = self._calculate_overall_score()
        
        return self.results
    
    def _analyze_data_availability(self) -> Dict:
        """Analiziraj dostupnost podataka."""
        sources = self.profile.data_sources
        
        if not sources:
            return {
                "total_sources": 0,
                "total_size_gb": 0,
                "access_distribution": {},
                "format_distribution": {},
                "avg_access_level": 0,
                "real_time_sources": 0,
                "api_enabled": 0,
                "availability_score": 0
            }
        
        # Distribucija pristupa
        access_dist = {}
        for level in DataAccessLevel:
            count = len([s for s in sources if s.access_level == level])
            if count > 0:
                access_dist[level.name] = count
        
        # Distribucija formata
        format_dist = {}
        for fmt in DataFormat:
            count = len([s for s in sources if s.format == fmt])
            if count > 0:
                format_dist[fmt.value] = count
        
        # Statistike
        total_size = sum(s.size_gb for s in sources)
        avg_access = sum(s.access_level.value for s in sources) / len(sources)
        real_time = len([s for s in sources if s.update_frequency == "real-time"])
        api_enabled = len([s for s in sources if s.has_api])
        
        # Score dostupnosti (0-100)
        availability_score = (
            (avg_access / 4) * 40 +  # Access level (max 40)
            (api_enabled / len(sources)) * 30 +  # API coverage (max 30)
            min(real_time / len(sources) * 30, 30)  # Real-time (max 30)
        )
        
        return {
            "total_sources": len(sources),
            "total_size_gb": total_size,
            "access_distribution": access_dist,
            "format_distribution": format_dist,
            "avg_access_level": avg_access,
            "real_time_sources": real_time,
            "api_enabled": api_enabled,
            "availability_score": availability_score
        }
    
    def _analyze_data_quality(self) -> Dict:
        """Analiziraj kvalitet podataka."""
        sources = self.profile.data_sources
        
        if not sources:
            return {
                "documented_sources": 0,
                "validated_sources": 0,
                "metadata_coverage": 0,
                "quality_score": 0
            }
        
        documented = len([s for s in sources if s.has_documentation])
        validated = len([s for s in sources if s.is_validated])
        with_metadata = len([s for s in sources if s.has_metadata])
        
        # Score kvaliteta
        quality_score = (
            (documented / len(sources)) * 35 +
            (validated / len(sources)) * 35 +
            (with_metadata / len(sources)) * 30
        ) * 100 / 100
        
        # Governance bonus
        if self.profile.has_data_quality_checks:
            quality_score += 10
        if self.profile.has_data_lineage:
            quality_score += 10
        if self.profile.has_data_catalog:
            quality_score += 10
        
        quality_score = min(quality_score, 100)
        
        return {
            "documented_sources": documented,
            "validated_sources": validated,
            "metadata_coverage": with_metadata,
            "has_data_catalog": self.profile.has_data_catalog,
            "has_data_lineage": self.profile.has_data_lineage,
            "has_quality_checks": self.profile.has_data_quality_checks,
            "quality_score": quality_score
        }
    
    def _analyze_interoperability(self) -> Dict:
        """Analiziraj nivo interoperabilnosti."""
        interfaces = self.profile.system_interfaces
        
        if not interfaces:
            return {
                "total_interfaces": 0,
                "standard_interfaces": 0,
                "protocol_distribution": {},
                "avg_latency_ms": 0,
                "avg_throughput_gbps": 0,
                "interoperability_level": InteroperabilityLevel.ISOLATED.name,
                "interoperability_score": 0
            }
        
        # Statistike interfejsa
        standard_count = len([i for i in interfaces if i.is_standard])
        
        # Distribucija protokola
        protocol_dist = {}
        for iface in interfaces:
            protocol_dist[iface.protocol] = protocol_dist.get(iface.protocol, 0) + 1
        
        # Performanse
        avg_latency = sum(i.latency_ms for i in interfaces) / len(interfaces)
        avg_throughput = sum(i.throughput_gbps for i in interfaces) / len(interfaces)
        
        # Odredi nivo interoperabilnosti
        standard_ratio = standard_count / len(interfaces)
        
        if standard_ratio >= 0.8 and self.profile.has_api_gateway:
            level = InteroperabilityLevel.SEAMLESS
        elif standard_ratio >= 0.6 and self.profile.uses_standard_protocols:
            level = InteroperabilityLevel.INTEGRATED
        elif standard_ratio >= 0.4:
            level = InteroperabilityLevel.STANDARDIZED
        elif len(interfaces) > 1:
            level = InteroperabilityLevel.POINT_TO_POINT
        else:
            level = InteroperabilityLevel.ISOLATED
        
        # Score interoperabilnosti
        interop_score = (
            (standard_ratio) * 40 +
            (level.value / 5) * 30 +
            (1 if self.profile.uses_standard_formats else 0) * 15 +
            (1 if self.profile.uses_standard_protocols else 0) * 15
        )
        
        return {
            "total_interfaces": len(interfaces),
            "standard_interfaces": standard_count,
            "protocol_distribution": protocol_dist,
            "avg_latency_ms": avg_latency,
            "avg_throughput_gbps": avg_throughput,
            "interoperability_level": level.name,
            "interoperability_score": interop_score * 100 / 100
        }
    
    def _analyze_integration_readiness(self) -> Dict:
        """Analiziraj spremnost za integraciju."""
        readiness_factors = {
            "has_etl_pipeline": self.profile.has_etl_pipeline,
            "has_data_warehouse": self.profile.has_data_warehouse,
            "has_api_gateway": self.profile.has_api_gateway,
            "has_data_catalog": self.profile.has_data_catalog,
            "has_backup_strategy": self.profile.has_backup_strategy,
            "has_data_standards_policy": self.profile.has_data_standards_policy
        }
        
        # Score
        score = sum(1 for v in readiness_factors.values() if v) / len(readiness_factors) * 100
        
        # Gaps
        gaps = [k.replace("has_", "").replace("_", " ").title() 
                for k, v in readiness_factors.items() if not v]
        
        return {
            "factors": readiness_factors,
            "readiness_score": score,
            "gaps": gaps
        }
    
    def _calculate_overall_score(self) -> float:
        """Izracunaj ukupni score."""
        weights = {
            "data_availability": 0.30,
            "data_quality": 0.25,
            "interoperability": 0.30,
            "integration_readiness": 0.15
        }
        
        score = (
            self.results["data_availability"].get("availability_score", 0) * weights["data_availability"] +
            self.results["data_quality"].get("quality_score", 0) * weights["data_quality"] +
            self.results["interoperability"].get("interoperability_score", 0) * weights["interoperability"] +
            self.results["integration_readiness"].get("readiness_score", 0) * weights["integration_readiness"]
        )
        
        return score
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generiši preporuke."""
        recommendations = []
        
        # Data availability recommendations
        avail = self.results.get("data_availability", {})
        if avail.get("api_enabled", 0) < avail.get("total_sources", 1) * 0.5:
            recommendations.append({
                "category": "data_availability",
                "priority": "HIGH",
                "title": "Implementacija API-ja za izvore podataka",
                "description": "Manje od 50% izvora podataka ima API pristup.",
                "action_items": [
                    "Identificirati kritične izvore podataka",
                    "Implementirati REST API za pristup",
                    "Dokumentovati API endpoint-e"
                ]
            })
        
        # Data quality recommendations
        quality = self.results.get("data_quality", {})
        if not self.profile.has_data_catalog:
            recommendations.append({
                "category": "data_quality",
                "priority": "HIGH",
                "title": "Uspostavljanje Data Catalog-a",
                "description": "Nedostaje centralni katalog podataka.",
                "action_items": [
                    "Odabrati alat za data catalog (Apache Atlas, Amundsen)",
                    "Inventarisati sve izvore podataka",
                    "Definisati metadata standarde"
                ]
            })
        
        if not self.profile.has_data_quality_checks:
            recommendations.append({
                "category": "data_quality",
                "priority": "MEDIUM",
                "title": "Implementacija Data Quality provjera",
                "description": "Nedostaju automatske provjere kvaliteta podataka.",
                "action_items": [
                    "Definisati pravila kvaliteta podataka",
                    "Implementirati automatske validacije",
                    "Postaviti alerting za anomalije"
                ]
            })
        
        # Interoperability recommendations
        interop = self.results.get("interoperability", {})
        if interop.get("interoperability_level") in ["ISOLATED", "POINT_TO_POINT"]:
            recommendations.append({
                "category": "interoperability",
                "priority": "CRITICAL",
                "title": "Standardizacija interfejsa",
                "description": "Nizak nivo interoperabilnosti sistema.",
                "action_items": [
                    "Usvojiti standardne protokole (REST, gRPC)",
                    "Definisati zajednicke formate podataka",
                    "Implementirati API gateway"
                ]
            })
        
        if not self.profile.uses_standard_formats:
            recommendations.append({
                "category": "interoperability",
                "priority": "MEDIUM",
                "title": "Prelazak na standardne formate podataka",
                "description": "Koriste se proprietarni formati podataka.",
                "action_items": [
                    "Identificirati proprietarne formate",
                    "Planirati migraciju na HDF5/NetCDF/Parquet",
                    "Implementirati konverzijske pipeline-e"
                ]
            })
        
        # Integration recommendations
        if not self.profile.has_etl_pipeline:
            recommendations.append({
                "category": "integration",
                "priority": "HIGH",
                "title": "Uspostavljanje ETL pipeline-a",
                "description": "Nedostaje automatizovani ETL proces.",
                "action_items": [
                    "Odabrati ETL alat (Apache Airflow, Luigi)",
                    "Definisati transformacijske workflow-e",
                    "Implementirati monitoring i logging"
                ]
            })
        
        if not self.profile.has_backup_strategy:
            recommendations.append({
                "category": "integration",
                "priority": "CRITICAL",
                "title": "Definisanje Backup strategije",
                "description": "Nedostaje strategija za backup podataka.",
                "action_items": [
                    "Definisati RPO i RTO ciljeve",
                    "Implementirati automatski backup",
                    "Testirati restore procedure"
                ]
            })
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generiši tekstualni izvještaj."""
        if not self.results:
            self.analyze()
        
        r = self.results
        avail = r["data_availability"]
        quality = r["data_quality"]
        interop = r["interoperability"]
        integ = r["integration_readiness"]
        
        report = f"""
===============================================================================
        ANALIZA DOSTUPNOSTI PODATAKA I INTEROPERABILNOSTI
                    {self.profile.organization_name}
===============================================================================

UKUPNI SCORE: {r['data_interop_score']:.1f}/100

-------------------------------------------------------------------------------
DOSTUPNOST PODATAKA (Score: {avail.get('availability_score', 0):.1f}/100)
-------------------------------------------------------------------------------

  Ukupno izvora podataka: {avail.get('total_sources', 0)}
  Ukupna velicina: {avail.get('total_size_gb', 0):.1f} GB
  Prosjecni nivo pristupa: {avail.get('avg_access_level', 0):.1f}/4
  Real-time izvori: {avail.get('real_time_sources', 0)}
  API-enabled izvori: {avail.get('api_enabled', 0)}

  Distribucija pristupa:
"""
        for level, count in avail.get("access_distribution", {}).items():
            report += f"    - {level}: {count}\n"
        
        report += f"""
  Distribucija formata:
"""
        for fmt, count in avail.get("format_distribution", {}).items():
            report += f"    - {fmt}: {count}\n"
        
        report += f"""
-------------------------------------------------------------------------------
KVALITET PODATAKA (Score: {quality.get('quality_score', 0):.1f}/100)
-------------------------------------------------------------------------------

  Dokumentovani izvori: {quality.get('documented_sources', 0)}
  Validirani izvori: {quality.get('validated_sources', 0)}
  Sa metapodacima: {quality.get('metadata_coverage', 0)}
  
  Data Governance:
    - Data Catalog: {'Da' if quality.get('has_data_catalog') else 'Ne'}
    - Data Lineage: {'Da' if quality.get('has_data_lineage') else 'Ne'}
    - Quality Checks: {'Da' if quality.get('has_quality_checks') else 'Ne'}

-------------------------------------------------------------------------------
INTEROPERABILNOST (Score: {interop.get('interoperability_score', 0):.1f}/100)
-------------------------------------------------------------------------------

  Nivo: {interop.get('interoperability_level', 'N/A')}
  Ukupno interfejsa: {interop.get('total_interfaces', 0)}
  Standardni interfejsi: {interop.get('standard_interfaces', 0)}
  Prosjecna latencija: {interop.get('avg_latency_ms', 0):.1f} ms
  Prosjecni throughput: {interop.get('avg_throughput_gbps', 0):.2f} Gbps

  Distribucija protokola:
"""
        for proto, count in interop.get("protocol_distribution", {}).items():
            report += f"    - {proto}: {count}\n"
        
        report += f"""
-------------------------------------------------------------------------------
SPREMNOST ZA INTEGRACIJU (Score: {integ.get('readiness_score', 0):.1f}/100)
-------------------------------------------------------------------------------

  Faktori:
    - ETL Pipeline: {'Da' if integ['factors'].get('has_etl_pipeline') else 'Ne'}
    - Data Warehouse: {'Da' if integ['factors'].get('has_data_warehouse') else 'Ne'}
    - API Gateway: {'Da' if integ['factors'].get('has_api_gateway') else 'Ne'}
    - Data Catalog: {'Da' if integ['factors'].get('has_data_catalog') else 'Ne'}
    - Backup Strategy: {'Da' if integ['factors'].get('has_backup_strategy') else 'Ne'}
    - Data Standards Policy: {'Da' if integ['factors'].get('has_data_standards_policy') else 'Ne'}

  Nedostaci (Gaps):
"""
        for gap in integ.get("gaps", []):
            report += f"    - {gap}\n"
        
        report += f"""
-------------------------------------------------------------------------------
PREPORUKE
-------------------------------------------------------------------------------
"""
        for rec in r.get("recommendations", []):
            report += f"""
  [{rec['priority']}] {rec['title']}
  Kategorija: {rec['category']}
  {rec['description']}
  Akcije:
"""
            for action in rec.get("action_items", []):
                report += f"    - {action}\n"
        
        report += """
===============================================================================
"""
        return report


def create_demo_data_interop_profile() -> DataInteroperabilityProfile:
    """Kreiraj demo profil za testiranje."""
    return DataInteroperabilityProfile(
        organization_name="Demo Organization",
        data_sources=[
            DataSource(
                name="Simulation Results DB",
                format=DataFormat.HDF5,
                access_level=DataAccessLevel.FULL,
                size_gb=500.0,
                update_frequency="daily",
                has_api=True,
                has_documentation=True,
                has_metadata=True,
                is_validated=True
            ),
            DataSource(
                name="Sensor Data",
                format=DataFormat.CSV,
                access_level=DataAccessLevel.REAL_TIME,
                size_gb=50.0,
                update_frequency="real-time",
                has_api=True,
                has_documentation=False,
                has_metadata=False,
                is_validated=False
            ),
            DataSource(
                name="Legacy Archive",
                format=DataFormat.PROPRIETARY,
                access_level=DataAccessLevel.LIMITED,
                size_gb=2000.0,
                update_frequency="static",
                has_api=False,
                has_documentation=False,
                has_metadata=False,
                is_validated=False
            ),
            DataSource(
                name="ML Training Data",
                format=DataFormat.PARQUET,
                access_level=DataAccessLevel.FULL,
                size_gb=100.0,
                update_frequency="hourly",
                has_api=True,
                has_documentation=True,
                has_metadata=True,
                is_validated=True
            )
        ],
        system_interfaces=[
            SystemInterface(
                name="HPC Job Submission API",
                protocol="REST",
                is_standard=True,
                has_authentication=True,
                has_versioning=True,
                latency_ms=5.0,
                throughput_gbps=1.0
            ),
            SystemInterface(
                name="Data Transfer Service",
                protocol="MPI-IO",
                is_standard=True,
                has_authentication=False,
                has_versioning=False,
                latency_ms=0.5,
                throughput_gbps=100.0
            ),
            SystemInterface(
                name="Legacy System Bridge",
                protocol="custom",
                is_standard=False,
                has_authentication=True,
                has_versioning=False,
                latency_ms=50.0,
                throughput_gbps=0.1
            )
        ],
        has_data_catalog=False,
        has_data_lineage=False,
        has_data_quality_checks=True,
        has_backup_strategy=True,
        uses_standard_formats=True,
        uses_standard_protocols=True,
        has_data_standards_policy=False,
        has_etl_pipeline=True,
        has_data_warehouse=False,
        has_api_gateway=False
    )


if __name__ == "__main__":
    # Demo
    profile = create_demo_data_interop_profile()
    analyzer = DataInteroperabilityAnalyzer(profile)
    results = analyzer.analyze()
    print(analyzer.generate_report())
