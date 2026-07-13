import type { Language } from './types'

const zh = {
  product: 'WaveST-MDT', subtitle: '肿瘤多学科会诊空间病理工作站', publicDemo: '公开演示',
  cases: '病例', atlas: '空间图谱', mdt: 'MDT会诊', report: '报告', audit: '证据审计',
  validation: '首个验证场景', currentCase: '当前病例', dataReady: '数据就绪',
  guide: '会诊路径', guideSteps: ['确认数据质量', '查看空间异质性', '复核高风险区域', '提交MDT意见', '签阅报告'],
  qc: '输入质量', registration: '空间配准', genes: '基因兼容性', reference: '参考图谱', passed: '通过',
  layer: '证据图层', dominant: '主要细胞状态', tumour: '肿瘤上皮', immune: '免疫浸润', stroma: '基质', uncertainty: '不确定性', niche: '空间niche',
  view3d: '3D组织', topView: '病理俯视', reset: '重置视角', evidence: 'spot证据', selectSpot: '点击空间点查看证据',
  cellComposition: '细胞组成', modality: '模态可靠性', image: '形态', expression: '表达', ref: '参考',
  trust: '复核等级', routine: '常规复核', focused: '重点复核', spot: '空间点',
  completeStep: '确认并继续', back: '上一步', next: '下一步',
  mdtTitle: 'MDT协作记录', addOpinion: '提交意见', opinionPlaceholder: '记录该区域的专业判断…',
  pathology: '病理科', oncology: '肿瘤科', molecular: '分子诊断', pending: '待会诊', reviewed: '已复核',
  reportTitle: '结构化会诊报告', print: '打印 / PDF', export: '导出证据JSON', sign: '完成签阅',
  auditTitle: '模型与证据追溯', modelVersion: '模型版本', dataset: '数据层级', checksum: '结果校验', retained: '审计记录完整',
  worklist: '病例工作列表', search: '检索病例', active: '分析完成', queued: '等待数据', importCase: '导入病例',
  researchOnly: '临床科研辅助，不替代病理诊断',
  realEvidence: '真实项目结果', demoCase: '去标识化演示病例',
}

const en: typeof zh = {
  product: 'WaveST-MDT', subtitle: 'Spatial pathology workstation for multidisciplinary oncology', publicDemo: 'PUBLIC DEMO',
  cases: 'Cases', atlas: 'Spatial atlas', mdt: 'MDT board', report: 'Report', audit: 'Evidence audit',
  validation: 'Initial validation context', currentCase: 'Current case', dataReady: 'Data ready',
  guide: 'Review pathway', guideSteps: ['Confirm input quality', 'Map spatial heterogeneity', 'Review high-risk regions', 'Submit MDT opinion', 'Sign report'],
  qc: 'Input quality', registration: 'Registration', genes: 'Gene compatibility', reference: 'Reference atlas', passed: 'Passed',
  layer: 'Evidence layer', dominant: 'Dominant cell state', tumour: 'Tumour epithelium', immune: 'Immune infiltration', stroma: 'Stroma', uncertainty: 'Uncertainty', niche: 'Spatial niche',
  view3d: '3D tissue', topView: 'Pathology top view', reset: 'Reset camera', evidence: 'Spot evidence', selectSpot: 'Select a spatial spot to inspect evidence',
  cellComposition: 'Cell composition', modality: 'Modality reliability', image: 'Morphology', expression: 'Expression', ref: 'Reference',
  trust: 'Review state', routine: 'Routine review', focused: 'Focused review', spot: 'Spatial spot',
  completeStep: 'Confirm and continue', back: 'Previous', next: 'Next',
  mdtTitle: 'MDT collaboration log', addOpinion: 'Submit opinion', opinionPlaceholder: 'Record a specialist interpretation for this region…',
  pathology: 'Pathology', oncology: 'Oncology', molecular: 'Molecular diagnostics', pending: 'Pending', reviewed: 'Reviewed',
  reportTitle: 'Structured MDT report', print: 'Print / PDF', export: 'Export evidence JSON', sign: 'Complete sign-off',
  auditTitle: 'Model and evidence provenance', modelVersion: 'Model version', dataset: 'Evidence tier', checksum: 'Result checksum', retained: 'Audit trail complete',
  worklist: 'Case worklist', search: 'Search cases', active: 'Analysis complete', queued: 'Awaiting data', importCase: 'Import case',
  researchOnly: 'Clinical research support; not a substitute for pathology diagnosis',
  realEvidence: 'Real project outputs', demoCase: 'De-identified demonstration case',
}

export function useCopy(language: Language) {
  return language === 'zh' ? zh : en
}
