/**
 * Shared TypeScript types for the AI Facultative Reinsurance System
 */

export enum DocumentType {
  PDF = "pdf",
  SCANNED_PDF = "scanned_pdf", 
  EMAIL = "email",
  EXCEL = "excel"
}

export interface Document {
  id: string;
  filename: string;
  document_type: DocumentType;
  file_path: string;
  upload_timestamp: string;
  processed: boolean;
  metadata: Record<string, any>;
}

export interface RiskParameters {
  asset_value?: number;
  coverage_limit?: number;
  asset_type?: string;
  location?: string;
  industry_sector?: string;
  construction_type?: string;
  occupancy?: string;
}

export interface FinancialData {
  revenue?: number;
  assets?: number;
  liabilities?: number;
  credit_rating?: string;
  financial_strength_rating?: string;
}

export interface LossEvent {
  date: string;
  amount: number;
  cause: string;
  description: string;
}

export interface RiskScore {
  overall_score: number;
  confidence: number;
  factors: Record<string, number>;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

export interface Recommendation {
  decision: "APPROVE" | "REJECT" | "CONDITIONAL";
  confidence: number;
  rationale: string;
  conditions: string[];
  premium_adjustment?: number;
  coverage_modifications: string[];
}

export interface Application {
  id: string;
  documents: Document[];
  risk_parameters: RiskParameters;
  financial_data: FinancialData;
  loss_history: LossEvent[];
  risk_analysis?: Record<string, any>;
  recommendation?: Recommendation;
  status: string;
  created_at: string;
  updated_at: string;
}