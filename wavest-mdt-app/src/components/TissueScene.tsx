import { Suspense, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber'
import { ContactShadows, Html, OrbitControls, useTexture } from '@react-three/drei'
import * as THREE from 'three'
import type { LayerKey, SpatialSpot, ViewMode } from '../types'

const CELL_PALETTE = [
  '#26a6a1', '#3b82a0', '#215c75', '#e1a13a', '#e36d5f', '#5cae77', '#8d6eaf',
  '#bd4a65', '#779e3b', '#d08545', '#a45179', '#d4b94f', '#54908a', '#8c9b70',
  '#875f49', '#d84d43', '#627f6d', '#6d8191', '#ad7658',
]
const NICHE_PALETTE = ['#d84d43', '#5c7f70', '#d59a36', '#2f8f86', '#a45473']

interface TissueSceneProps {
  spots: SpatialSpot[]
  cellTypes: string[]
  layer: LayerKey
  viewMode: ViewMode
  selectedSpot: SpatialSpot | null
  onSelectSpot: (spot: SpatialSpot) => void
  cameraResetKey: number
}

function CameraRig({ viewMode, resetKey }: { viewMode: ViewMode; resetKey: number }) {
  const { camera } = useThree()
  useEffect(() => {
    const target = viewMode === 'top' ? new THREE.Vector3(0, 11.8, 0.01) : new THREE.Vector3(8.8, 7.2, 8.6)
    camera.position.copy(target)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()
  }, [camera, viewMode, resetKey])
  return null
}

function TissuePlane() {
  const texture = useTexture(`${import.meta.env.BASE_URL}assets/tissue-lowres.png`)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 8
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[11.6, 8.7, 1, 1]} />
        <meshStandardMaterial map={texture} color="#ffffff" roughness={0.94} transparent opacity={0.88} />
      </mesh>
      <mesh position={[0, -0.035, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[11.72, 8.82]} />
        <meshStandardMaterial color="#d8dbd5" roughness={1} />
      </mesh>
    </group>
  )
}

function SpatialInstances({ spots, cellTypes, layer, onSelectSpot }: Omit<TissueSceneProps, 'viewMode' | 'selectedSpot' | 'cameraResetKey'>) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const maxUncertainty = useMemo(() => Math.max(...spots.map((spot) => spot.uncertainty), 0.001), [spots])

  const groups = useMemo(() => {
    const has = (pattern: RegExp) => cellTypes.map((name, index) => pattern.test(name) ? index : -1).filter((index) => index >= 0)
    return {
      tumour: has(/Tumou?r|DCIS/i),
      immune: has(/Cells|DCs|Macrophage|Mast/i),
      stroma: has(/Stromal|Perivascular|Endothelial|Myoepi/i),
    }
  }, [cellTypes])

  const values = useMemo(() => spots.map((spot) => {
    if (layer === 'uncertainty') return Math.min(1, spot.uncertainty / maxUncertainty)
    if (layer === 'niche') return 0.64 + spot.niche * 0.06
    if (layer === 'tumour' || layer === 'immune' || layer === 'stroma') {
      return groups[layer].reduce((sum, index) => sum + spot.values[index], 0)
    }
    return spot.values[spot.dominant]
  }), [groups, layer, maxUncertainty, spots])

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const low = new THREE.Color('#ced6cf')
    const uncertaintyHigh = new THREE.Color('#d84d43')
    const uncertaintyLow = new THREE.Color('#2c9c8e')
    spots.forEach((spot, index) => {
      const value = values[index]
      const columnHeight = 0.07 + value * (layer === 'uncertainty' ? 0.72 : 0.9)
      dummy.position.set(spot.x * 5.45, 0.035 + columnHeight / 2, spot.y * 4.05)
      const scale = layer === 'dominant' ? 0.042 + value * 0.026 : 0.044 + value * 0.034
      dummy.scale.set(scale, columnHeight, scale)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(index, dummy.matrix)

      let color = low.clone()
      if (layer === 'dominant') color.set(CELL_PALETTE[spot.dominant % CELL_PALETTE.length])
      else if (layer === 'niche') color.set(NICHE_PALETTE[spot.niche % NICHE_PALETTE.length])
      else if (layer === 'uncertainty') color.lerpColors(uncertaintyLow, uncertaintyHigh, value)
      else {
        const target = layer === 'tumour' ? '#d84d43' : layer === 'immune' ? '#2b7e9a' : '#4f8a68'
        color.lerpColors(low, new THREE.Color(target), Math.min(1, value * 1.35))
      }
      mesh.setColorAt(index, color)
    })
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [dummy, layer, spots, values])

  const select = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    if (event.instanceId !== undefined) onSelectSpot(spots[event.instanceId])
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, spots.length]}
      castShadow
      onClick={select}
      onPointerOver={() => { document.body.style.cursor = 'crosshair' }}
      onPointerOut={() => { document.body.style.cursor = 'default' }}
    >
      <cylinderGeometry args={[1, 1, 1, 8]} />
      <meshStandardMaterial roughness={0.52} metalness={0} vertexColors flatShading />
    </instancedMesh>
  )
}

function SelectedMarker({ spot }: { spot: SpatialSpot }) {
  return (
    <group position={[spot.x * 5.45, 0.15, spot.y * 4.05]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.16, 0.22, 32]} />
        <meshBasicMaterial color="#fff7db" side={THREE.DoubleSide} transparent opacity={0.96} />
      </mesh>
      <Html center position={[0, 0.56, 0]} distanceFactor={9}>
        <div className="scene-spot-label">{spot.id.slice(0, 8)}</div>
      </Html>
    </group>
  )
}

export default function TissueScene(props: TissueSceneProps) {
  return (
    <div className="tissue-canvas" data-testid="tissue-canvas">
      <Canvas camera={{ position: [8.8, 7.2, 8.6], fov: 39, near: 0.1, far: 100 }} dpr={[1, 1.65]} gl={{ preserveDrawingBuffer: true, antialias: true }} shadows>
        <color attach="background" args={['#12201e']} />
        <fog attach="fog" args={['#12201e', 14, 28]} />
        <hemisphereLight intensity={1.7} color="#fff4dc" groundColor="#24534c" />
        <ambientLight intensity={1.8} />
        <directionalLight position={[5, 10, 3]} intensity={2.5} color="#fff1d5" castShadow />
        <directionalLight position={[-6, 4, -5]} intensity={1.15} color="#9dd9d4" />
        <Suspense fallback={null}>
          <TissuePlane />
        </Suspense>
        <SpatialInstances spots={props.spots} cellTypes={props.cellTypes} layer={props.layer} onSelectSpot={props.onSelectSpot} />
        {props.selectedSpot && <SelectedMarker spot={props.selectedSpot} />}
        <ContactShadows position={[0, -0.04, 0]} opacity={0.28} scale={16} blur={2.8} far={8} />
        <gridHelper args={[22, 44, '#37514d', '#213734']} position={[0, -0.07, 0]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={5.5} maxDistance={24} maxPolarAngle={Math.PI / 2.04} />
        <CameraRig viewMode={props.viewMode} resetKey={props.cameraResetKey} />
      </Canvas>
    </div>
  )
}
