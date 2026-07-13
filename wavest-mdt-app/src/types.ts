export type Language = 'zh' | 'en'
export type ViewMode = '3d' | 'top'
export type LayerKey = 'dominant' | 'tumour' | 'immune' | 'stroma' | 'uncertainty' | 'niche'
export type WorkspaceView = 'atlas' | 'cases' | 'mdt' | 'report' | 'audit'

export interface SpatialSpot {
  id: string
  x: number
  y: number
  values: number[]
  dominant: number
  uncertainty: number
  gates: [number, number, number]
  niche: number
  nicheLabel: string
}

export interface SpatialDemo {
  case: {
    id: string
    title: string
    titleZh: string
    specimen: string
    specimenZh: string
    validationContext: string
    validationContextZh: string
    platform: string
    model: string
  }
  cellTypes: string[]
  aggregate: {
    spots: number
    cellStates: number
    supervisedSpots: number
    typedCells: number
    meanProportions: number[]
    uncertaintyCorrelation: number
    boundaryJumpRatio: number
    calibrationBinCorrelation: number
  }
  spots: SpatialSpot[]
}
