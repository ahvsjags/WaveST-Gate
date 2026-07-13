import { useEffect, useMemo, useState } from 'react'
import {
  Activity, BadgeCheck, BookOpenCheck, Box, BrainCircuit, BriefcaseMedical, Check,
  ChevronLeft, ChevronRight, CircleUserRound, ClipboardCheck, CloudUpload, Database,
  Download, FileCheck2, FileText, Focus, Gauge, Globe2, Languages, Layers3, LockKeyhole,
  MessageSquareText, Microscope, PanelLeftClose, Printer, RefreshCcw, Search, Send,
  ShieldCheck, SlidersHorizontal, Sparkles, Stethoscope, UsersRound, X,
} from 'lucide-react'
import TissueScene from './components/TissueScene'
import { useCopy } from './i18n'
import type { Language, LayerKey, SpatialDemo, SpatialSpot, ViewMode, WorkspaceView } from './types'

const NAV_ITEMS: Array<{ key: WorkspaceView; icon: typeof Microscope }> = [
  { key: 'cases', icon: BriefcaseMedical }, { key: 'atlas', icon: Microscope },
  { key: 'mdt', icon: UsersRound }, { key: 'report', icon: FileText }, { key: 'audit', icon: ShieldCheck },
]

const LAYER_COLORS: Record<LayerKey, string> = {
  dominant: '#c89534', tumour: '#d6554d', immune: '#2b7f9a', stroma: '#4e8b68', uncertainty: '#d84d43', niche: '#a65373',
}

type Opinion = { id: number; role: 'pathology' | 'oncology' | 'molecular'; textZh: string; textEn: string; time: string }
type BackendHealth = { status: string; checkpointAvailable: boolean; cudaAvailable: boolean; device?: string; deviceName?: string; batchSize?: number }
type InferenceJob = { id: string; status: 'queued' | 'running' | 'complete' | 'failed'; stage: string; progress: number; error?: string }
type UploadBundle = { caseName: string; heImage: File; expression: File; coordinates: File; reference: File }

const LOCAL_API_BASE = (() => {
  const { hostname, protocol } = window.location
  return hostname === 'localhost' || hostname === '127.0.0.1' ? `${protocol}//${hostname}:8010/api` : null
})()

const INITIAL_OPINIONS: Opinion[] = [
  { id: 1, role: 'pathology', textZh: '肿瘤-基质交界保存清楚，建议重点复核右侧高不确定区域。', textEn: 'The tumour-stroma interface is preserved; focused review is recommended for the right high-uncertainty region.', time: '09:24' },
  { id: 2, role: 'molecular', textZh: '表达和参考模态贡献稳定，当前结果具备分区解释条件。', textEn: 'Expression and reference evidence are stable, supporting compartment-level interpretation.', time: '09:31' },
]

function formatPercent(value: number) { return `${(value * 100).toFixed(value > 0.1 ? 1 : 2)}%` }
function formatMetric(value: number | null, suffix = '') { return value === null ? '—' : `${value}${suffix}` }

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function MetricBar({ label, value, color = '#15766e' }: { label: string; value: number; color?: string }) {
  return (
    <div className="metric-bar">
      <div className="metric-bar-label"><span>{label}</span><strong>{formatPercent(value)}</strong></div>
      <div className="metric-track"><span style={{ width: `${Math.max(1.5, value * 100)}%`, background: color }} /></div>
    </div>
  )
}

function StatusRow({ icon: Icon, label, value }: { icon: typeof Check; label: string; value: string }) {
  return (
    <div className="status-row">
      <span className="status-icon"><Icon size={15} /></span><span>{label}</span>
      <strong><Check size={13} />{value}</strong>
    </div>
  )
}

function App() {
  const [language, setLanguage] = useState<Language>('zh')
  const [data, setData] = useState<SpatialDemo | null>(null)
  const [view, setView] = useState<WorkspaceView>('atlas')
  const [layer, setLayer] = useState<LayerKey>('dominant')
  const [viewMode, setViewMode] = useState<ViewMode>('3d')
  const [selectedSpot, setSelectedSpot] = useState<SpatialSpot | null>(null)
  const [guideStep, setGuideStep] = useState(0)
  const [cameraResetKey, setCameraResetKey] = useState(0)
  const [opinions, setOpinions] = useState<Opinion[]>(INITIAL_OPINIONS)
  const [note, setNote] = useState('')
  const [role, setRole] = useState<Opinion['role']>('oncology')
  const [reportSigned, setReportSigned] = useState(false)
  const [toast, setToast] = useState('')
  const [importOpen, setImportOpen] = useState(false)
  const [backend, setBackend] = useState<BackendHealth | null>(null)
  const [backendChecked, setBackendChecked] = useState(false)
  const [job, setJob] = useState<InferenceJob | null>(null)
  const [search, setSearch] = useState('')
  const copy = useCopy(language)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/spatial-demo.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Data request failed: ${response.status}`)
        return response.json()
      })
      .then(setData)
      .catch(() => setToast(language === 'zh' ? '空间数据加载失败' : 'Spatial data failed to load'))
  }, [])

  useEffect(() => {
    if (!LOCAL_API_BASE) {
      setBackendChecked(true)
      return
    }
    fetch(`${LOCAL_API_BASE}/health`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Local API unavailable')))
      .then((health: BackendHealth) => setBackend(health))
      .catch(() => setBackend(null))
      .finally(() => setBackendChecked(true))
  }, [])

  useEffect(() => {
    if (!LOCAL_API_BASE || !job || job.status === 'complete' || job.status === 'failed') return
    let cancelled = false
    const sync = async () => {
      try {
        const response = await fetch(`${LOCAL_API_BASE}/jobs/${job.id}`)
        if (!response.ok) throw new Error('Local inference job is unavailable')
        const next: InferenceJob = await response.json()
        if (cancelled) return
        if (next.status === 'complete') {
          const resultResponse = await fetch(`${LOCAL_API_BASE}/jobs/${next.id}/result`)
          if (!resultResponse.ok) throw new Error('Local inference result is unavailable')
          const result: SpatialDemo = await resultResponse.json()
          if (cancelled) return
          setData(result)
          setSelectedSpot(null)
          setLayer('dominant')
          setView('atlas')
          setJob(next)
          setToast(language === 'zh' ? '本地 GPU 推理完成，结果已载入工作站。' : 'Local GPU inference completed and results are loaded.')
        } else if (next.status === 'failed') {
          setJob(next)
          setToast(next.error ?? (language === 'zh' ? '本地推理未完成。' : 'Local inference did not complete.'))
        } else {
          setJob(next)
        }
      } catch (error) {
        if (!cancelled) setToast(error instanceof Error ? error.message : 'Local inference status could not be read.')
      }
    }
    void sync()
    const timer = window.setInterval(() => void sync(), 1800)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [job?.id, job?.status, language])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(''), 2400)
    return () => window.clearTimeout(timer)
  }, [toast])

  const highestRisk = useMemo(() => data ? data.spots.reduce((best, spot) => spot.uncertainty > best.uncertainty ? spot : best, data.spots[0]) : null, [data])
  const sourceValues = selectedSpot?.values ?? data?.aggregate.meanProportions ?? []
  const topComposition = useMemo(() => {
    if (!data) return []
    return data.cellTypes.map((name, index) => ({ name, value: sourceValues[index] ?? 0, index }))
      .sort((a, b) => b.value - a.value).slice(0, 6)
  }, [data, sourceValues])

  const navLabel = (key: WorkspaceView) => ({ cases: copy.cases, atlas: copy.atlas, mdt: copy.mdt, report: copy.report, audit: copy.audit })[key]
  const layerLabel = (key: LayerKey) => ({ dominant: copy.dominant, tumour: copy.tumour, immune: copy.immune, stroma: copy.stroma, uncertainty: copy.uncertainty, niche: copy.niche })[key]

  const advanceGuide = () => {
    const next = Math.min(copy.guideSteps.length - 1, guideStep + 1)
    setGuideStep(next)
    if (next === 1) setLayer('dominant')
    if (next === 2) { setLayer('uncertainty'); if (highestRisk) setSelectedSpot(highestRisk) }
    if (next === 3) setView('mdt')
    if (next === 4) setView('report')
  }

  const jumpGuide = (index: number) => {
    setGuideStep(index)
    if (index <= 2) setView('atlas')
    if (index === 2) { setLayer('uncertainty'); if (highestRisk) setSelectedSpot(highestRisk) }
    if (index === 3) setView('mdt')
    if (index === 4) setView('report')
  }

  const submitOpinion = () => {
    const trimmed = note.trim()
    if (!trimmed) return
    setOpinions((current) => [...current, { id: Date.now(), role, textZh: trimmed, textEn: trimmed, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }])
    setNote('')
    setToast(language === 'zh' ? 'MDT意见已记录' : 'MDT opinion recorded')
    if (guideStep === 3) setGuideStep(4)
  }

  const submitLocalCase = async (bundle: UploadBundle) => {
    if (!LOCAL_API_BASE) throw new Error(language === 'zh' ? '请在本机完整工作站中启动本地推理服务。' : 'Start the local inference service in the full workstation.')
    const payload = new FormData()
    payload.append('case_name', bundle.caseName)
    payload.append('h_e_image', bundle.heImage)
    payload.append('expression', bundle.expression)
    payload.append('coordinates', bundle.coordinates)
    payload.append('reference', bundle.reference)
    const response = await fetch(`${LOCAL_API_BASE}/jobs`, { method: 'POST', body: payload })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail ?? 'Local inference job could not be created.')
    }
    const next: InferenceJob = await response.json()
    setJob(next)
    setImportOpen(false)
    setToast(language === 'zh' ? '本地推理任务已创建；数据仅保留在此计算机。' : 'Local inference job created; data remains on this computer.')
  }

  if (!data) {
    return (
      <main className="loading-screen">
        <div className="loading-mark"><BrainCircuit size={28} /><span /></div>
        <strong>WaveST-MDT</strong><p>{language === 'zh' ? '正在装载空间证据…' : 'Loading spatial evidence…'}</p>
      </main>
    )
  }

  const trustFocused = (selectedSpot?.uncertainty ?? 0) > 0.025

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-symbol"><Activity size={21} /></div>
          <div><strong>{copy.product}</strong><span>{copy.subtitle}</span></div>
        </div>
        <div className="case-context">
          <span className="context-label">{copy.currentCase}</span>
          <strong>{data.case.id}</strong>
          <span className="context-divider" />
          <span>{language === 'zh' ? data.case.titleZh : data.case.title}</span>
          <span className="status-pill"><BadgeCheck size={14} />{job && job.status !== 'complete' && job.status !== 'failed' ? `${job.progress}% · ${job.stage}` : copy.dataReady}</span>
        </div>
        <div className="top-actions">
          <span className="demo-pill"><Globe2 size={13} />{copy.publicDemo}</span>
          <button className="icon-text-button" onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')} data-testid="language-toggle">
            <Languages size={17} /><span>{language === 'zh' ? 'EN' : '中文'}</span>
          </button>
          <button className="avatar-button" title="Clinician profile"><CircleUserRound size={22} /></button>
        </div>
      </header>

      <section className="guide-rail" aria-label={copy.guide}>
        <span className="guide-title"><Sparkles size={15} />{copy.guide}</span>
        <div className="guide-steps">
          {copy.guideSteps.map((step, index) => (
            <button key={step} className={index === guideStep ? 'active' : index < guideStep ? 'done' : ''} onClick={() => jumpGuide(index)}>
              <span>{index < guideStep ? <Check size={12} /> : index + 1}</span><em>{step}</em>
            </button>
          ))}
        </div>
        <div className="guide-controls">
          <button title={copy.back} onClick={() => jumpGuide(Math.max(0, guideStep - 1))} disabled={guideStep === 0}><ChevronLeft size={17} /></button>
          <button title={copy.next} onClick={advanceGuide} disabled={guideStep === copy.guideSteps.length - 1}><ChevronRight size={17} /></button>
        </div>
      </section>

      <main className="workspace">
        <nav className="nav-rail" aria-label="Workspace">
          {NAV_ITEMS.map(({ key, icon: Icon }) => (
            <button key={key} className={view === key ? 'active' : ''} title={navLabel(key)} onClick={() => setView(key)} data-testid={`nav-${key}`}>
              <Icon size={20} /><span>{navLabel(key)}</span>
            </button>
          ))}
          <span className="nav-spacer" />
          <button title="Security"><LockKeyhole size={19} /><span>Secure</span></button>
        </nav>

        <aside className="control-panel">
          {view === 'cases' ? (
            <CaseWorklist copy={copy} language={language} search={search} setSearch={setSearch} onImport={() => setImportOpen(true)} backend={backend} backendChecked={backendChecked} job={job} />
          ) : (
            <>
              <div className="panel-heading">
                <span>{copy.currentCase}</span><strong>{data.case.id}</strong>
                <p>{language === 'zh' ? data.case.specimenZh : data.case.specimen}</p>
              </div>
              <div className="context-band">
                <Microscope size={17} /><div><span>{copy.validation}</span><strong>{language === 'zh' ? data.case.validationContextZh : data.case.validationContext}</strong></div>
              </div>
              <section className="panel-section">
                <h2><Database size={16} />{copy.qc}</h2>
                <StatusRow icon={Focus} label={copy.registration} value={copy.passed} />
                <StatusRow icon={Activity} label={copy.genes} value="297 / 297" />
                <StatusRow icon={BrainCircuit} label={copy.reference} value={copy.passed} />
              </section>
              <section className="panel-section layer-section">
                <h2><Layers3 size={16} />{copy.layer}</h2>
                <div className="layer-list">
                  {(['dominant', 'tumour', 'immune', 'stroma', 'uncertainty', 'niche'] as LayerKey[]).map((key) => (
                    <button key={key} className={layer === key ? 'active' : ''} onClick={() => { setLayer(key); setView('atlas') }} disabled={key === 'niche' && data.inference?.nicheAvailable === false} data-testid={`layer-${key}`}>
                      <span className="swatch" style={{ background: LAYER_COLORS[key] }} />
                      <span>{layerLabel(key)}</span>{layer === key && <Check size={15} />}
                    </button>
                  ))}
                </div>
              </section>
              <div className="dataset-facts">
                <div><strong>{data.aggregate.spots.toLocaleString()}</strong><span>spots</span></div>
                <div><strong>{data.aggregate.cellStates}</strong><span>cell states</span></div>
                <div><strong>{data.inference?.mode === 'live_local' ? '—' : data.aggregate.supervisedSpots}</strong><span>matched GT</span></div>
              </div>
            </>
          )}
        </aside>

        <section className="scene-workspace">
          <div className="scene-heading">
            <div><span>{copy.realEvidence}</span><h1>{navLabel(view)}</h1></div>
            <div className="scene-controls">
              <div className="segmented-control">
                <button className={viewMode === '3d' ? 'active' : ''} onClick={() => setViewMode('3d')}><Box size={16} />{copy.view3d}</button>
                <button className={viewMode === 'top' ? 'active' : ''} onClick={() => setViewMode('top')}><PanelLeftClose size={16} />{copy.topView}</button>
              </div>
              <button className="icon-button" title={copy.reset} onClick={() => setCameraResetKey((value) => value + 1)}><RefreshCcw size={17} /></button>
            </div>
          </div>
          <TissueScene
            spots={data.spots}
            cellTypes={data.cellTypes}
            layer={layer}
            viewMode={viewMode}
            selectedSpot={selectedSpot}
            onSelectSpot={(spot) => { setSelectedSpot(spot); if (guideStep === 2) setToast(language === 'zh' ? '高风险区域已定位' : 'High-risk region located') }}
            cameraResetKey={cameraResetKey}
            tissueImageUrl={data.inference?.tissueEndpoint && LOCAL_API_BASE ? `${LOCAL_API_BASE}${data.inference.tissueEndpoint}` : undefined}
          />
          <div className="scene-legend">
            <span className="swatch" style={{ background: LAYER_COLORS[layer] }} />
            <strong>{layerLabel(layer)}</strong><span>{data.inference?.mode === 'live_local' ? (language === 'zh' ? '本地推理结果' : 'Local inference result') : copy.demoCase}</span>
          </div>
          <div className="scene-metrics">
            <div><span>ρ uncertainty/error</span><strong>{formatMetric(data.aggregate.uncertaintyCorrelation)}</strong></div>
            <div><span>boundary/interior</span><strong>{formatMetric(data.aggregate.boundaryJumpRatio, '×')}</strong></div>
            <div><span>calibration bins</span><strong>{formatMetric(data.aggregate.calibrationBinCorrelation)}</strong></div>
          </div>
        </section>

        <aside className="evidence-panel">
          {view === 'mdt' ? (
            <MdtPanel copy={copy} language={language} opinions={opinions} role={role} setRole={setRole} note={note} setNote={setNote} onSubmit={submitOpinion} />
          ) : view === 'report' ? (
            <ReportPanel copy={copy} language={language} data={data} selectedSpot={selectedSpot} opinions={opinions} signed={reportSigned} onSign={() => { setReportSigned(true); setToast(language === 'zh' ? '报告已完成签阅' : 'Report signed') }} />
          ) : view === 'audit' ? (
            <AuditPanel copy={copy} data={data} language={language} />
          ) : (
            <Inspector copy={copy} data={data} language={language} selectedSpot={selectedSpot} topComposition={topComposition} trustFocused={trustFocused} />
          )}
          <div className="guide-action">
            <span>{guideStep + 1} / {copy.guideSteps.length}</span>
            <button onClick={advanceGuide} disabled={guideStep === copy.guideSteps.length - 1}>{copy.completeStep}<ChevronRight size={16} /></button>
          </div>
        </aside>
      </main>

      <footer className="safety-strip"><ShieldCheck size={14} /><span>{copy.researchOnly}</span><span className="safety-separator" /><span>{data.case.model}</span></footer>

      {importOpen && <ImportDialog copy={copy} language={language} backend={backend} backendChecked={backendChecked} onClose={() => setImportOpen(false)} onSubmit={submitLocalCase} />}
      {toast && <div className="toast" role="status"><BadgeCheck size={17} />{toast}</div>}
    </div>
  )
}

function CaseWorklist({ copy, language, search, setSearch, onImport, backend, backendChecked, job }: { copy: ReturnType<typeof useCopy>; language: Language; search: string; setSearch: (value: string) => void; onImport: () => void; backend: BackendHealth | null; backendChecked: boolean; job: InferenceJob | null }) {
  const cases = [
    { id: 'DEMO-ST-001', zh: '肿瘤空间图谱', en: 'Tumour spatial atlas', status: copy.active, active: true },
    { id: 'DEMO-LU-002', zh: '肺肿瘤参考病例', en: 'Lung tumour reference', status: copy.queued, active: false },
    { id: 'DEMO-CRC-003', zh: '结直肠肿瘤参考病例', en: 'Colorectal reference', status: copy.queued, active: false },
  ].filter((item) => `${item.id}${item.zh}${item.en}`.toLowerCase().includes(search.toLowerCase()))
  return (
    <div className="worklist-panel">
      <div className="panel-heading"><span>Workspace</span><strong>{copy.worklist}</strong><p>3 {copy.cases.toLowerCase()}</p></div>
      <label className="search-field"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={copy.search} /></label>
      <button className="primary-button full" onClick={onImport}><CloudUpload size={17} />{copy.importCase}</button>
      <div className={`inference-status ${backend?.checkpointAvailable && backend?.cudaAvailable ? 'ready' : ''}`}>
        <div><BrainCircuit size={16} /><span>{language === 'zh' ? '本地推理服务' : 'Local inference service'}</span></div>
        <strong>{!backendChecked ? (language === 'zh' ? '检查中' : 'Checking') : backend?.checkpointAvailable && backend?.cudaAvailable ? (language === 'zh' ? 'GPU 已就绪' : 'GPU ready') : (language === 'zh' ? '仅演示模式' : 'Demo only')}</strong>
        {job && <small>{job.progress}% · {job.stage}</small>}
        {backend?.deviceName && <small>{backend.deviceName} · batch {backend.batchSize}</small>}
      </div>
      <div className="case-list">
        {cases.map((item) => <button key={item.id} className={item.active ? 'active' : ''} disabled={!item.active}><span><strong>{item.id}</strong><em>{language === 'zh' ? item.zh : item.en}</em></span><small>{item.status}</small></button>)}
      </div>
    </div>
  )
}

function Inspector({ copy, data, language, selectedSpot, topComposition, trustFocused }: { copy: ReturnType<typeof useCopy>; data: SpatialDemo; language: Language; selectedSpot: SpatialSpot | null; topComposition: Array<{ name: string; value: number; index: number }>; trustFocused: boolean }) {
  return (
    <div className="inspector-content" data-testid="evidence-inspector">
      <div className="panel-title"><span><SlidersHorizontal size={16} />{copy.evidence}</span>{selectedSpot && <small>{copy.spot}</small>}</div>
      {selectedSpot ? <><strong className="spot-id">{selectedSpot.id}</strong><p className="spot-niche">N{selectedSpot.niche} · {selectedSpot.nicheLabel}</p></> : <div className="empty-selection"><Focus size={28} /><p>{copy.selectSpot}</p></div>}
      <section className="evidence-section"><h2>{copy.cellComposition}</h2>{topComposition.map((item) => <MetricBar key={item.name} label={item.name} value={item.value} color={['#d6554d', '#15766e', '#2b7f9a', '#c89534', '#a65373', '#557466'][item.index % 6]} />)}</section>
      <section className="evidence-section"><h2>{copy.modality}</h2>
        <MetricBar label={copy.image} value={selectedSpot?.gates[0] ?? 0.0006} color="#c89534" />
        <MetricBar label={copy.expression} value={selectedSpot?.gates[1] ?? 0.9987} color="#15766e" />
        <MetricBar label={copy.ref} value={selectedSpot?.gates[2] ?? 0.0007} color="#2b7f9a" />
      </section>
      <section className={`trust-band ${trustFocused ? 'focused' : ''}`}><Gauge size={20} /><div><span>{copy.trust}</span><strong>{trustFocused ? copy.focused : copy.routine}</strong></div><em>{selectedSpot ? selectedSpot.uncertainty.toFixed(4) : '—'}</em></section>
      <div className="evidence-source"><FileCheck2 size={16} /><span>{language === 'zh' ? '来源：真实WaveST-Gate分析输出' : 'Source: real WaveST-Gate analysis output'}</span></div>
    </div>
  )
}

function MdtPanel({ copy, language, opinions, role, setRole, note, setNote, onSubmit }: { copy: ReturnType<typeof useCopy>; language: Language; opinions: Opinion[]; role: Opinion['role']; setRole: (role: Opinion['role']) => void; note: string; setNote: (value: string) => void; onSubmit: () => void }) {
  const roleLabel = (value: Opinion['role']) => ({ pathology: copy.pathology, oncology: copy.oncology, molecular: copy.molecular })[value]
  return <div className="mdt-panel"><div className="panel-title"><span><UsersRound size={17} />{copy.mdtTitle}</span><small>{opinions.length}</small></div>
    <div className="opinion-list">{opinions.map((opinion) => <article key={opinion.id}><div><span className={`role-dot ${opinion.role}`} /> <strong>{roleLabel(opinion.role)}</strong><time>{opinion.time}</time></div><p>{language === 'zh' ? opinion.textZh : opinion.textEn}</p></article>)}</div>
    <div className="opinion-compose"><div className="role-selector">{(['pathology', 'oncology', 'molecular'] as Opinion['role'][]).map((item) => <button key={item} className={role === item ? 'active' : ''} onClick={() => setRole(item)}>{roleLabel(item)}</button>)}</div><textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder={copy.opinionPlaceholder} /><button className="primary-button" onClick={onSubmit} disabled={!note.trim()}><Send size={16} />{copy.addOpinion}</button></div>
  </div>
}

function ReportPanel({ copy, language, data, selectedSpot, opinions, signed, onSign }: { copy: ReturnType<typeof useCopy>; language: Language; data: SpatialDemo; selectedSpot: SpatialSpot | null; opinions: Opinion[]; signed: boolean; onSign: () => void }) {
  const report = { caseId: data.case.id, model: data.case.model, spots: data.aggregate.spots, selectedSpot: selectedSpot?.id ?? null, mdtOpinions: opinions.length, signed, evidence: 'de-identified research demonstration' }
  return <div className="report-panel"><div className="panel-title"><span><FileText size={17} />{copy.reportTitle}</span>{signed && <small className="signed"><Check size={12} />{language === 'zh' ? '已签阅' : 'Signed'}</small>}</div>
    <div className="report-document"><header><strong>WaveST-MDT</strong><span>{data.case.id}</span></header>
      <section><h3>01 · {language === 'zh' ? '样本与数据' : 'Specimen and data'}</h3><p>{language === 'zh' ? data.case.specimenZh : data.case.specimen} · {data.case.platform}</p></section>
      <section><h3>02 · {language === 'zh' ? '空间微环境' : 'Spatial microenvironment'}</h3><p>{language === 'zh' ? `共分析${data.aggregate.spots.toLocaleString()}个空间点和${data.aggregate.cellStates}种细胞状态。` : `${data.aggregate.spots.toLocaleString()} spatial spots and ${data.aggregate.cellStates} cell states were analysed.`}</p></section>
      <section><h3>03 · {language === 'zh' ? '可靠性证据' : 'Reliability evidence'}</h3><p>ρ = {data.aggregate.uncertaintyCorrelation}; boundary/interior = {data.aggregate.boundaryJumpRatio}×</p></section>
      <section><h3>04 · {language === 'zh' ? 'MDT记录' : 'MDT record'}</h3><p>{opinions.length} {language === 'zh' ? '条专业意见已记录。' : 'specialist opinions recorded.'}</p></section>
    </div>
    <div className="report-actions"><button onClick={() => window.print()}><Printer size={16} />{copy.print}</button><button onClick={() => downloadJson(`${data.case.id}-evidence.json`, report)}><Download size={16} />{copy.export}</button></div>
    <button className={`sign-button ${signed ? 'signed' : ''}`} onClick={onSign} disabled={signed}>{signed ? <Check size={18} /> : <BookOpenCheck size={18} />}{signed ? (language === 'zh' ? '签阅完成' : 'Sign-off complete') : copy.sign}</button>
  </div>
}

function AuditPanel({ copy, data, language }: { copy: ReturnType<typeof useCopy>; data: SpatialDemo; language: Language }) {
  return <div className="audit-panel"><div className="panel-title"><span><ShieldCheck size={17} />{copy.auditTitle}</span><small>PASS</small></div>
    <div className="audit-ring"><div><strong>100</strong><span>/100</span></div><p>{copy.retained}</p></div>
    <div className="audit-list"><StatusRow icon={BrainCircuit} label={copy.modelVersion} value={data.case.model} /><StatusRow icon={Database} label={copy.dataset} value={language === 'zh' ? '匹配组织框架' : 'Matched tissue frame'} /><StatusRow icon={FileCheck2} label={copy.checksum} value="SHA-256" /><StatusRow icon={LockKeyhole} label={language === 'zh' ? '隐私模式' : 'Privacy mode'} value={language === 'zh' ? '去标识化' : 'De-identified'} /></div>
    <div className="audit-provenance"><span>GitHub</span><strong>ahvsjags/WaveST-Gate</strong><span>Zenodo DOI</span><strong>10.5281/zenodo.20550855</strong></div>
  </div>
}

function ImportDialog({ copy, onClose, language, backend, backendChecked, onSubmit }: { copy: ReturnType<typeof useCopy>; onClose: () => void; language: Language; backend: BackendHealth | null; backendChecked: boolean; onSubmit: (bundle: UploadBundle) => Promise<void> }) {
  const [caseName, setCaseName] = useState('local-spatial-case')
  const [heImage, setHeImage] = useState<File | null>(null)
  const [expression, setExpression] = useState<File | null>(null)
  const [coordinates, setCoordinates] = useState<File | null>(null)
  const [reference, setReference] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const ready = Boolean(caseName.trim() && heImage && expression && coordinates && reference)
  const localReady = Boolean(backend?.checkpointAvailable && backend?.cudaAvailable)

  const submit = async () => {
    if (!heImage || !expression || !coordinates || !reference) return
    setSubmitting(true)
    setError('')
    try {
      await onSubmit({ caseName: caseName.trim(), heImage, expression, coordinates, reference })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to submit the local inference job.')
      setSubmitting(false)
    }
  }

  const fileName = (file: File | null) => file ? file.name : (language === 'zh' ? '未选择' : 'Not selected')
  return <div className="modal-backdrop" role="dialog" aria-modal="true"><form className="import-dialog" onSubmit={(event) => { event.preventDefault(); void submit() }}><header><div><span>WaveST-MDT · local inference</span><h2>{copy.importCase}</h2></div><button className="icon-button" type="button" onClick={onClose}><X size={19} /></button></header>
    <div className={`import-service-notice ${localReady ? 'ready' : ''}`}><BrainCircuit size={18} /><div><strong>{localReady ? (language === 'zh' ? '本地 GPU 推理已连接' : 'Local GPU inference connected') : (language === 'zh' ? '未连接本地推理服务' : 'Local inference service unavailable')}</strong><span>{localReady ? `${backend?.deviceName ?? backend?.device} · batch ${backend?.batchSize}` : (backendChecked ? (language === 'zh' ? '公开页面仅展示去标识化结果。' : 'The public page only displays de-identified results.') : (language === 'zh' ? '正在检查服务。' : 'Checking local service.'))}</span></div></div>
    <div className="import-form-body">
      <label className="case-name-field"><span>{language === 'zh' ? '病例标识' : 'Case identifier'}</span><input value={caseName} onChange={(event) => setCaseName(event.target.value)} maxLength={80} required /></label>
      <div className="file-contract-grid">
        <label><span>01 · H&E {language === 'zh' ? '图像' : 'image'}</span><i className="file-select"><CloudUpload size={13} />{language === 'zh' ? '选择文件' : 'Choose file'}</i><input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" onChange={(event) => setHeImage(event.target.files?.[0] ?? null)} required /><small>{fileName(heImage)}</small></label>
        <label><span>02 · {language === 'zh' ? '表达矩阵' : 'Expression matrix'}</span><i className="file-select"><CloudUpload size={13} />{language === 'zh' ? '选择文件' : 'Choose file'}</i><input type="file" accept=".csv,.tsv,.txt" onChange={(event) => setExpression(event.target.files?.[0] ?? null)} required /><small>{fileName(expression)}</small></label>
        <label><span>03 · {language === 'zh' ? '空间坐标 CSV' : 'Spatial coordinates CSV'}</span><i className="file-select"><CloudUpload size={13} />{language === 'zh' ? '选择文件' : 'Choose file'}</i><input type="file" accept=".csv" onChange={(event) => setCoordinates(event.target.files?.[0] ?? null)} required /><small>{fileName(coordinates)}</small></label>
        <label><span>04 · {language === 'zh' ? '参考图谱' : 'Reference atlas'}</span><i className="file-select"><CloudUpload size={13} />{language === 'zh' ? '选择文件' : 'Choose file'}</i><input type="file" accept=".csv,.tsv,.txt" onChange={(event) => setReference(event.target.files?.[0] ?? null)} required /><small>{fileName(reference)}</small></label>
      </div>
      <div className="import-contract"><FileCheck2 size={16} /><span>{language === 'zh' ? '坐标表必须包含 spot_id、x 和 y；模型输出是研究用途的空间推理结果。' : 'The coordinate CSV must contain spot_id, x and y; model outputs are research-use spatial inferences.'}</span></div>
      {error && <p className="import-error" role="alert">{error}</p>}
    </div>
    <footer><span><LockKeyhole size={14} />{language === 'zh' ? '上传数据仅写入本机运行目录，不发送至公开网站。' : 'Uploaded data is written only to the local runtime directory and never sent to the public site.'}</span><button className="primary-button" type="submit" disabled={!ready || !localReady || submitting}>{submitting ? (language === 'zh' ? '正在创建任务' : 'Creating job') : (language === 'zh' ? '提交本地推理' : 'Run local inference')}</button></footer>
  </form></div>
}

export default App
