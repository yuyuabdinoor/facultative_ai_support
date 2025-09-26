/**
 * Shared constants for the AI Facultative Reinsurance System
 */

export const API_ENDPOINTS = {
  DOCUMENTS: '/api/v1/documents',
  APPLICATIONS: '/api/v1/applications',
  RISK_ANALYSIS: '/api/v1/risk-analysis',
  RECOMMENDATIONS: '/api/v1/recommendations',
  HEALTH: '/health'
} as const;

export const FILE_TYPES = {
  PDF: 'application/pdf',
  EXCEL: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  MSG: 'application/vnd.ms-outlook'
} as const;

export const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

export const RISK_LEVELS = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM', 
  HIGH: 'HIGH',
  CRITICAL: 'CRITICAL'
} as const;

export const DECISION_TYPES = {
  APPROVE: 'APPROVE',
  REJECT: 'REJECT',
  CONDITIONAL: 'CONDITIONAL'
} as const;