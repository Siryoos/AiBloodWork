"""Label taxonomies and clinical classifications for bloodwork AI."""

from enum import Enum
from typing import Dict, List, Optional


class AnemiaType(str, Enum):
    """Anemia classification types."""
    
    NORMAL = "normal"
    IRON_DEFICIENCY = "iron_deficiency"
    THALASSEMIA_TRAIT = "thalassemia_trait"
    B12_DEFICIENCY = "b12_deficiency"
    FOLATE_DEFICIENCY = "folate_deficiency"
    ANEMIA_OF_CHRONIC_DISEASE = "anemia_of_chronic_disease"
    HEMOLYTIC = "hemolytic"
    APLASTIC = "aplastic"
    UNKNOWN = "unknown"


class ThyroidDysfunctionType(str, Enum):
    """Thyroid dysfunction classification."""
    
    NORMAL = "normal"
    HYPOTHYROIDISM = "hypothyroidism"
    HYPERTHYROIDISM = "hyperthyroidism"
    SUBCLINICAL_HYPOTHYROIDISM = "subclinical_hypothyroidism"
    SUBCLINICAL_HYPERTHYROIDISM = "subclinical_hyperthyroidism"
    UNKNOWN = "unknown"


class DICStage(str, Enum):
    """Disseminated Intravascular Coagulation stages."""
    
    NORMAL = "normal"
    EARLY = "early"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class MetabolicSyndromeCriteria(str, Enum):
    """Metabolic syndrome diagnostic criteria."""
    
    NORMAL = "normal"
    METABOLIC_SYNDROME = "metabolic_syndrome"
    UNKNOWN = "unknown"


class CVDRiskLevel(str, Enum):
    """Cardiovascular disease risk levels."""
    
    LOW = "low"  # < 10%
    INTERMEDIATE = "intermediate"  # 10-20%
    HIGH = "high"  # > 20%
    UNKNOWN = "unknown"


class B12DeficiencyLevel(str, Enum):
    """Vitamin B12 deficiency levels."""
    
    NORMAL = "normal"  # > 300 pg/mL
    BORDERLINE = "borderline"  # 200-300 pg/mL
    DEFICIENT = "deficient"  # < 200 pg/mL
    UNKNOWN = "unknown"


class FerritinLevel(str, Enum):
    """Ferritin levels."""
    
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    UNKNOWN = "unknown"


# Clinical thresholds and criteria
CLINICAL_THRESHOLDS = {
    "anemia": {
        "hb_male": 13.0,  # g/dL
        "hb_female": 12.0,  # g/dL
        "hct_male": 39.0,  # %
        "hct_female": 36.0,  # %
    },
    "thyroid": {
        "tsh_normal_min": 0.4,  # mIU/L
        "tsh_normal_max": 4.0,  # mIU/L
        "tsh_subclinical_min": 4.0,  # mIU/L
        "tsh_subclinical_max": 10.0,  # mIU/L
        "ft4_normal_min": 0.8,  # ng/dL
        "ft4_normal_max": 1.8,  # ng/dL
    },
    "metabolic_syndrome": {
        "waist_circumference_male": 102,  # cm
        "waist_circumference_female": 88,  # cm
        "triglycerides": 150,  # mg/dL
        "hdl_male": 40,  # mg/dL
        "hdl_female": 50,  # mg/dL
        "systolic_bp": 130,  # mmHg
        "diastolic_bp": 85,  # mmHg
        "glucose": 100,  # mg/dL
    },
    "cvd_risk": {
        "low_threshold": 0.1,  # 10%
        "high_threshold": 0.2,  # 20%
    },
    "dic": {
        "d_dimer_threshold": 500,  # ng/mL
        "pt_threshold": 15,  # seconds
        "aptt_threshold": 40,  # seconds
        "fibrinogen_threshold": 200,  # mg/dL
        "platelet_threshold": 100,  # K/μL
    },
    "b12_deficiency": {
        "deficient_threshold": 200,  # pg/mL
        "borderline_threshold": 300,  # pg/mL
    },
    "ferritin": {
        "low_male": 30,  # ng/mL
        "low_female": 15,  # ng/mL
        "high_threshold": 300,  # ng/mL
    },
}


def classify_anemia(cbc_data: Dict, age: int, sex: str) -> AnemiaType:
    """
    Classify anemia type based on CBC values.
    
    Args:
        cbc_data: Dictionary containing CBC values
        age: Patient age
        sex: Patient sex (M/F)
        
    Returns:
        Anemia classification
    """
    hb = cbc_data.get("hb")
    mcv = cbc_data.get("mcv")
    rdw = cbc_data.get("rdw")
    
    if hb is None:
        return AnemiaType.UNKNOWN
    
    # Get appropriate threshold based on sex
    hb_threshold = CLINICAL_THRESHOLDS["anemia"]["hb_female"] if sex.upper() in ["F", "FEMALE"] else CLINICAL_THRESHOLDS["anemia"]["hb_male"]
    
    if hb >= hb_threshold:
        return AnemiaType.NORMAL
    
    # Microcytic anemia (MCV < 80)
    if mcv is not None and mcv < 80:
        if rdw is not None and rdw > 15:
            return AnemiaType.IRON_DEFICIENCY
        else:
            return AnemiaType.THALASSEMIA_TRAIT
    
    # Macrocytic anemia (MCV > 100)
    if mcv is not None and mcv > 100:
        return AnemiaType.B12_DEFICIENCY
    
    # Normocytic anemia
    return AnemiaType.ANEMIA_OF_CHRONIC_DISEASE


def classify_thyroid_dysfunction(tsh: float, ft4: Optional[float] = None) -> ThyroidDysfunctionType:
    """
    Classify thyroid dysfunction based on TSH and FT4 values.
    
    Args:
        tsh: TSH value (mIU/L)
        ft4: Free T4 value (ng/dL), optional
        
    Returns:
        Thyroid dysfunction classification
    """
    if tsh is None:
        return ThyroidDysfunctionType.UNKNOWN
    
    thresholds = CLINICAL_THRESHOLDS["thyroid"]
    
    if tsh < thresholds["tsh_normal_min"]:
        if ft4 is not None and ft4 > thresholds["ft4_normal_max"]:
            return ThyroidDysfunctionType.HYPERTHYROIDISM
        else:
            return ThyroidDysfunctionType.SUBCLINICAL_HYPERTHYROIDISM
    elif tsh > thresholds["tsh_normal_max"]:
        if tsh <= thresholds["tsh_subclinical_max"]:
            return ThyroidDysfunctionType.SUBCLINICAL_HYPOTHYROIDISM
        else:
            if ft4 is not None and ft4 < thresholds["ft4_normal_min"]:
                return ThyroidDysfunctionType.HYPOTHYROIDISM
            else:
                return ThyroidDysfunctionType.SUBCLINICAL_HYPOTHYROIDISM
    else:
        return ThyroidDysfunctionType.NORMAL


def classify_metabolic_syndrome(
    waist_circumference: Optional[float] = None,
    triglycerides: Optional[float] = None,
    hdl: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    glucose: Optional[float] = None,
    sex: Optional[str] = None
) -> MetabolicSyndromeCriteria:
    """
    Classify metabolic syndrome based on diagnostic criteria.
    
    Args:
        waist_circumference: Waist circumference (cm)
        triglycerides: Triglycerides (mg/dL)
        hdl: HDL cholesterol (mg/dL)
        systolic_bp: Systolic blood pressure (mmHg)
        diastolic_bp: Diastolic blood pressure (mmHg)
        glucose: Fasting glucose (mg/dL)
        sex: Patient sex (M/F)
        
    Returns:
        Metabolic syndrome classification
    """
    criteria = CLINICAL_THRESHOLDS["metabolic_syndrome"]
    criteria_met = 0
    
    # Waist circumference
    if waist_circumference is not None:
        threshold = criteria["waist_circumference_female"] if sex and sex.upper() in ["F", "FEMALE"] else criteria["waist_circumference_male"]
        if waist_circumference >= threshold:
            criteria_met += 1
    
    # Triglycerides
    if triglycerides is not None and triglycerides >= criteria["triglycerides"]:
        criteria_met += 1
    
    # HDL cholesterol
    if hdl is not None:
        threshold = criteria["hdl_female"] if sex and sex.upper() in ["F", "FEMALE"] else criteria["hdl_male"]
        if hdl < threshold:
            criteria_met += 1
    
    # Blood pressure
    if systolic_bp is not None and systolic_bp >= criteria["systolic_bp"]:
        criteria_met += 1
    elif diastolic_bp is not None and diastolic_bp >= criteria["diastolic_bp"]:
        criteria_met += 1
    
    # Glucose
    if glucose is not None and glucose >= criteria["glucose"]:
        criteria_met += 1
    
    if criteria_met >= 3:
        return MetabolicSyndromeCriteria.METABOLIC_SYNDROME
    else:
        return MetabolicSyndromeCriteria.NORMAL


def classify_cvd_risk(risk_score: float) -> CVDRiskLevel:
    """
    Classify CVD risk level based on risk score.
    
    Args:
        risk_score: CVD risk score (0-1)
        
    Returns:
        CVD risk level classification
    """
    thresholds = CLINICAL_THRESHOLDS["cvd_risk"]
    
    if risk_score < thresholds["low_threshold"]:
        return CVDRiskLevel.LOW
    elif risk_score < thresholds["high_threshold"]:
        return CVDRiskLevel.INTERMEDIATE
    else:
        return CVDRiskLevel.HIGH


def classify_dic_stage(
    d_dimer: Optional[float] = None,
    pt: Optional[float] = None,
    aptt: Optional[float] = None,
    fibrinogen: Optional[float] = None,
    platelets: Optional[float] = None
) -> DICStage:
    """
    Classify DIC stage based on coagulation parameters.
    
    Args:
        d_dimer: D-dimer (ng/mL)
        pt: Prothrombin time (seconds)
        aptt: Activated partial thromboplastin time (seconds)
        fibrinogen: Fibrinogen (mg/dL)
        platelets: Platelet count (K/μL)
        
    Returns:
        DIC stage classification
    """
    criteria = CLINICAL_THRESHOLDS["dic"]
    abnormal_count = 0
    
    if d_dimer is not None and d_dimer > criteria["d_dimer_threshold"]:
        abnormal_count += 1
    if pt is not None and pt > criteria["pt_threshold"]:
        abnormal_count += 1
    if aptt is not None and aptt > criteria["aptt_threshold"]:
        abnormal_count += 1
    if fibrinogen is not None and fibrinogen < criteria["fibrinogen_threshold"]:
        abnormal_count += 1
    if platelets is not None and platelets < criteria["platelet_threshold"]:
        abnormal_count += 1
    
    if abnormal_count == 0:
        return DICStage.NORMAL
    elif abnormal_count <= 2:
        return DICStage.EARLY
    elif abnormal_count <= 3:
        return DICStage.MODERATE
    else:
        return DICStage.SEVERE


def classify_b12_deficiency(b12: float) -> B12DeficiencyLevel:
    """
    Classify B12 deficiency level.
    
    Args:
        b12: Vitamin B12 level (pg/mL)
        
    Returns:
        B12 deficiency level classification
    """
    thresholds = CLINICAL_THRESHOLDS["b12_deficiency"]
    
    if b12 < thresholds["deficient_threshold"]:
        return B12DeficiencyLevel.DEFICIENT
    elif b12 < thresholds["borderline_threshold"]:
        return B12DeficiencyLevel.BORDERLINE
    else:
        return B12DeficiencyLevel.NORMAL


def classify_ferritin_level(ferritin: float, sex: str) -> FerritinLevel:
    """
    Classify ferritin level.
    
    Args:
        ferritin: Ferritin level (ng/mL)
        sex: Patient sex (M/F)
        
    Returns:
        Ferritin level classification
    """
    thresholds = CLINICAL_THRESHOLDS["ferritin"]
    
    low_threshold = thresholds["low_female"] if sex.upper() in ["F", "FEMALE"] else thresholds["low_male"]
    
    if ferritin < low_threshold:
        return FerritinLevel.LOW
    elif ferritin > thresholds["high_threshold"]:
        return FerritinLevel.HIGH
    else:
        return FerritinLevel.NORMAL
