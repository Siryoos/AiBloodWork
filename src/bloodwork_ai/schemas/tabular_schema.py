"""Pydantic schemas for tabular lab data validation."""

from typing import Optional, Union
from pydantic import BaseModel, Field, validator


class CBCSchema(BaseModel):
    """Complete Blood Count schema."""
    
    # Basic counts
    wbc: Optional[float] = Field(None, ge=0, le=100, description="White blood cell count (K/μL)")
    rbc: Optional[float] = Field(None, ge=0, le=10, description="Red blood cell count (M/μL)")
    hb: Optional[float] = Field(None, ge=0, le=20, description="Hemoglobin (g/dL)")
    hct: Optional[float] = Field(None, ge=0, le=60, description="Hematocrit (%)")
    plt: Optional[float] = Field(None, ge=0, le=1000, description="Platelet count (K/μL)")
    
    # Red cell indices
    mcv: Optional[float] = Field(None, ge=50, le=120, description="Mean corpuscular volume (fL)")
    mch: Optional[float] = Field(None, ge=20, le=40, description="Mean corpuscular hemoglobin (pg)")
    mchc: Optional[float] = Field(None, ge=30, le=40, description="Mean corpuscular hemoglobin concentration (g/dL)")
    rdw: Optional[float] = Field(None, ge=10, le=25, description="Red cell distribution width (%)")
    
    # White cell differential
    neut_pct: Optional[float] = Field(None, ge=0, le=100, description="Neutrophil percentage (%)")
    lymph_pct: Optional[float] = Field(None, ge=0, le=100, description="Lymphocyte percentage (%)")
    mono_pct: Optional[float] = Field(None, ge=0, le=100, description="Monocyte percentage (%)")
    eos_pct: Optional[float] = Field(None, ge=0, le=100, description="Eosinophil percentage (%)")
    baso_pct: Optional[float] = Field(None, ge=0, le=100, description="Basophil percentage (%)")
    
    @validator('neut_pct', 'lymph_pct', 'mono_pct', 'eos_pct', 'baso_pct')
    def validate_differential_percentages(cls, v, values):
        """Validate that differential percentages sum to approximately 100%."""
        if v is not None:
            # Get all differential values
            diffs = [
                values.get('neut_pct', 0) or 0,
                values.get('lymph_pct', 0) or 0,
                values.get('mono_pct', 0) or 0,
                values.get('eos_pct', 0) or 0,
                values.get('baso_pct', 0) or 0,
            ]
            total = sum(d for d in diffs if d is not None)
            if total > 0 and abs(total - 100) > 5:  # Allow 5% tolerance
                raise ValueError(f"Differential percentages sum to {total}%, should be ~100%")
        return v


class CMPSchema(BaseModel):
    """Comprehensive Metabolic Panel schema."""
    
    # Electrolytes
    na: Optional[float] = Field(None, ge=120, le=160, description="Sodium (mEq/L)")
    k: Optional[float] = Field(None, ge=2.0, le=8.0, description="Potassium (mEq/L)")
    cl: Optional[float] = Field(None, ge=80, le=120, description="Chloride (mEq/L)")
    co2: Optional[float] = Field(None, ge=15, le=35, description="CO2 (mEq/L)")
    
    # Kidney function
    bun: Optional[float] = Field(None, ge=0, le=50, description="Blood urea nitrogen (mg/dL)")
    cr: Optional[float] = Field(None, ge=0, le=5, description="Creatinine (mg/dL)")
    
    # Glucose
    glucose: Optional[float] = Field(None, ge=50, le=500, description="Glucose (mg/dL)")
    
    # Liver function
    ast: Optional[float] = Field(None, ge=0, le=200, description="AST (U/L)")
    alt: Optional[float] = Field(None, ge=0, le=200, description="ALT (U/L)")
    alp: Optional[float] = Field(None, ge=0, le=500, description="Alkaline phosphatase (U/L)")
    bilirubin: Optional[float] = Field(None, ge=0, le=10, description="Total bilirubin (mg/dL)")
    
    # Proteins
    protein: Optional[float] = Field(None, ge=4, le=10, description="Total protein (g/dL)")
    albumin: Optional[float] = Field(None, ge=2, le=6, description="Albumin (g/dL)")
    
    # Calcium
    ca: Optional[float] = Field(None, ge=7, le=12, description="Calcium (mg/dL)")


class LipidSchema(BaseModel):
    """Lipid panel schema."""
    
    total_chol: Optional[float] = Field(None, ge=50, le=500, description="Total cholesterol (mg/dL)")
    ldl: Optional[float] = Field(None, ge=0, le=300, description="LDL cholesterol (mg/dL)")
    hdl: Optional[float] = Field(None, ge=10, le=150, description="HDL cholesterol (mg/dL)")
    triglycerides: Optional[float] = Field(None, ge=0, le=1000, description="Triglycerides (mg/dL)")
    
    @validator('ldl')
    def validate_ldl_calculation(cls, v, values):
        """Validate LDL calculation if all values are present."""
        if v is not None:
            total_chol = values.get('total_chol')
            hdl = values.get('hdl')
            triglycerides = values.get('triglycerides')
            
            if all(x is not None for x in [total_chol, hdl, triglycerides]):
                # Friedewald equation: LDL = Total - HDL - (Triglycerides/5)
                calculated_ldl = total_chol - hdl - (triglycerides / 5)
                if abs(v - calculated_ldl) > 10:  # Allow 10 mg/dL tolerance
                    raise ValueError(f"LDL value {v} doesn't match calculated value {calculated_ldl}")
        return v


class ThyroidSchema(BaseModel):
    """Thyroid function panel schema."""
    
    tsh: Optional[float] = Field(None, ge=0, le=100, description="TSH (mIU/L)")
    ft4: Optional[float] = Field(None, ge=0, le=5, description="Free T4 (ng/dL)")
    ft3: Optional[float] = Field(None, ge=0, le=10, description="Free T3 (pg/mL)")
    
    @validator('tsh', 'ft4', 'ft3')
    def validate_thyroid_values(cls, v):
        """Validate thyroid function values are reasonable."""
        if v is not None and v <= 0:
            raise ValueError("Thyroid values must be positive")
        return v


class CoagSchema(BaseModel):
    """Coagulation panel schema."""
    
    pt: Optional[float] = Field(None, ge=8, le=50, description="Prothrombin time (seconds)")
    inr: Optional[float] = Field(None, ge=0.5, le=10, description="International normalized ratio")
    aptt: Optional[float] = Field(None, ge=15, le=100, description="Activated partial thromboplastin time (seconds)")
    d_dimer: Optional[float] = Field(None, ge=0, le=10000, description="D-dimer (ng/mL)")
    fibrinogen: Optional[float] = Field(None, ge=100, le=800, description="Fibrinogen (mg/dL)")
    platelets: Optional[float] = Field(None, ge=0, le=1000, description="Platelet count (K/μL)")


class VitaminsIronSchema(BaseModel):
    """Vitamins and iron panel schema."""
    
    # Iron studies
    ferritin: Optional[float] = Field(None, ge=0, le=1000, description="Ferritin (ng/mL)")
    iron: Optional[float] = Field(None, ge=0, le=300, description="Serum iron (μg/dL)")
    tibc: Optional[float] = Field(None, ge=100, le=600, description="Total iron binding capacity (μg/dL)")
    transferrin: Optional[float] = Field(None, ge=150, le=400, description="Transferrin (mg/dL)")
    transferrin_sat: Optional[float] = Field(None, ge=0, le=100, description="Transferrin saturation (%)")
    
    # Vitamins
    b12: Optional[float] = Field(None, ge=0, le=2000, description="Vitamin B12 (pg/mL)")
    folate: Optional[float] = Field(None, ge=0, le=50, description="Folate (ng/mL)")
    
    # Inflammatory marker
    crp: Optional[float] = Field(None, ge=0, le=50, description="C-reactive protein (mg/L)")
    
    @validator('transferrin_sat')
    def validate_transferrin_saturation(cls, v, values):
        """Validate transferrin saturation calculation."""
        if v is not None:
            iron = values.get('iron')
            tibc = values.get('tibc')
            
            if iron is not None and tibc is not None and tibc > 0:
                calculated_sat = (iron / tibc) * 100
                if abs(v - calculated_sat) > 5:  # Allow 5% tolerance
                    raise ValueError(f"Transferrin saturation {v}% doesn't match calculated {calculated_sat}%")
        return v


class LabPanelSchema(BaseModel):
    """Combined lab panel schema."""
    
    # Patient information
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    date: Optional[str] = Field(None, description="Lab date (YYYY-MM-DD)")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age (years)")
    sex: Optional[str] = Field(None, pattern="^(M|F|Male|Female)$", description="Patient sex")
    
    # Lab panels
    cbc: Optional[CBCSchema] = Field(None, description="Complete Blood Count")
    cmp: Optional[CMPSchema] = Field(None, description="Comprehensive Metabolic Panel")
    lipid: Optional[LipidSchema] = Field(None, description="Lipid Panel")
    thyroid: Optional[ThyroidSchema] = Field(None, description="Thyroid Panel")
    coag: Optional[CoagSchema] = Field(None, description="Coagulation Panel")
    vitamins_iron: Optional[VitaminsIronSchema] = Field(None, description="Vitamins and Iron Panel")
    
    # Clinical outcomes (for training)
    anemia_type: Optional[str] = Field(None, description="Anemia classification")
    thyroid_dysfunction: Optional[str] = Field(None, description="Thyroid dysfunction type")
    metabolic_syndrome: Optional[bool] = Field(None, description="Metabolic syndrome diagnosis")
    cvd_risk: Optional[float] = Field(None, ge=0, le=1, description="CVD risk score")
    dic_stage: Optional[str] = Field(None, description="DIC stage")
    b12_deficiency: Optional[bool] = Field(None, description="B12 deficiency")
    ferritin_low: Optional[bool] = Field(None, description="Low ferritin")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"
