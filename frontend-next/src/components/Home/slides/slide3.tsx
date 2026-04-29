/**
 * Slide 3 — "Shape every detail."
 * Animation: Parameter knobs/sliders animate, waveform morphs smoothly in real-time.
 * Loops every 8 seconds.
 */
import React, { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { addPropertyControls, ControlType } from "./framer-stub"
import { C, Icons, glassPanel, SlideFrame } from "./shared"

// ── Seeded RNG ──
function seededRandom(seed: number) {
    let s = seed
    return () => {
        s = (s * 16807 + 0) % 2147483647
        return (s - 1) / 2147483646
    }
}

// ── Pre-generate smoothed amplitude arrays for each phase ──
const BAR_W = 4
const BAR_GAP = 3
const BAR_STEP = BAR_W + BAR_GAP
const BAR_R = BAR_W / 2
const WAVE_W = 500
const WAVE_H = 120
const BAR_COUNT = Math.floor(WAVE_W / BAR_STEP)

function buildAmps(seed: number): number[] {
    const rng = seededRandom(seed)
    const raw: number[] = []
    for (let i = 0; i < BAR_COUNT; i++) {
        const pos = i / BAR_COUNT
        const env = 0.3 + 0.7 * Math.sin(pos * Math.PI)
        const burst = rng() > 0.88 ? 1.3 : 1
        raw.push(rng() * env * burst)
    }
    // Moving average smooth
    const smoothed: number[] = []
    const radius = 3
    for (let i = 0; i < raw.length; i++) {
        let sum = 0,
            count = 0
        for (
            let j = Math.max(0, i - radius);
            j <= Math.min(raw.length - 1, i + radius);
            j++
        ) {
            sum += raw[j]
            count++
        }
        smoothed.push(sum / count)
    }
    return smoothed
}

// Seed 777 = before/base, then one per phase
const phaseAmps = [
    buildAmps(777), // base / before
    buildAmps(877), // after phase 0
    buildAmps(977), // after phase 1
    buildAmps(1077), // after phase 2
]

// ── Canvas waveform that morphs between shapes ──
function MorphingWaveform({
    phaseIndex,
    color,
    beforeColor,
}: {
    phaseIndex: number
    color: string
    beforeColor: string
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const animRef = useRef<number>(0)
    const currentAmps = useRef<number[]>([...phaseAmps[0]])
    const targetAmps = useRef<number[]>(phaseAmps[1])
    const prevPhase = useRef(-1)

    useEffect(() => {
        // When phase changes, set new target
        if (phaseIndex !== prevPhase.current) {
            prevPhase.current = phaseIndex
            targetAmps.current = phaseAmps[phaseIndex + 1] // +1 because 0 is base
        }
    }, [phaseIndex])

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")!
        const dpr = window.devicePixelRatio || 1
        canvas.width = WAVE_W * dpr
        canvas.height = WAVE_H * dpr
        canvas.style.width = `${WAVE_W}px`
        canvas.style.height = `${WAVE_H}px`
        ctx.scale(dpr, dpr)

        const mid = WAVE_H / 2
        const maxAmp = WAVE_H * 0.42

        const drawBars = (amps: number[], fillStyle: string) => {
            ctx.fillStyle = fillStyle
            for (let i = 0; i < BAR_COUNT; i++) {
                const x = i * BAR_STEP
                const h = amps[i] * maxAmp
                if (h < 1) continue
                const r = Math.min(BAR_R, h)
                ctx.beginPath()
                ctx.roundRect(x, mid - h, BAR_W, h * 2, r)
                ctx.fill()
            }
        }

        const animate = () => {
            const cur = currentAmps.current
            const tgt = targetAmps.current

            // Lerp each bar toward target — slow, smooth glide
            const speed = 0.035
            for (let i = 0; i < BAR_COUNT; i++) {
                cur[i] += (tgt[i] - cur[i]) * speed
            }

            ctx.clearRect(0, 0, WAVE_W, WAVE_H)

            // Before waveform (dimmed, static base)
            drawBars(phaseAmps[0], beforeColor)

            // After waveform (morphing)
            drawBars(cur, color)

            animRef.current = requestAnimationFrame(animate)
        }
        animRef.current = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(animRef.current)
    }, [color, beforeColor])

    return (
        <canvas
            ref={canvasRef}
            style={{ display: "block", width: WAVE_W, height: WAVE_H }}
        />
    )
}

const GLOW = "50,200,80"
const CYCLE = 8000

interface ParamDef {
    label: string
    value: number
    target: number
    unit: string
    color: string
}

const paramSets: ParamDef[][] = [
    // Phase 1: tighten space only
    [
        {
            label: "Space",
            value: 72,
            target: 12,
            unit: "%",
            color: "100,180,255",
        },
        {
            label: "Clarity",
            value: 45,
            target: 45,
            unit: "%",
            color: "80,220,140",
        },
        {
            label: "Density",
            value: 40,
            target: 40,
            unit: "%",
            color: "180,120,255",
        },
        {
            label: "Brightness",
            value: 50,
            target: 50,
            unit: "%",
            color: "255,180,60",
        },
        {
            label: "Pitch",
            value: 0,
            target: 0,
            unit: "st",
            color: "200,160,80",
        },
    ],
    // Phase 2: boost clarity only
    [
        {
            label: "Space",
            value: 12,
            target: 12,
            unit: "%",
            color: "100,180,255",
        },
        {
            label: "Clarity",
            value: 45,
            target: 92,
            unit: "%",
            color: "80,220,140",
        },
        {
            label: "Density",
            value: 40,
            target: 40,
            unit: "%",
            color: "180,120,255",
        },
        {
            label: "Brightness",
            value: 50,
            target: 50,
            unit: "%",
            color: "255,180,60",
        },
        {
            label: "Pitch",
            value: 0,
            target: 0,
            unit: "st",
            color: "200,160,80",
        },
    ],
    // Phase 3: increase brightness only
    [
        {
            label: "Space",
            value: 12,
            target: 12,
            unit: "%",
            color: "100,180,255",
        },
        {
            label: "Clarity",
            value: 92,
            target: 92,
            unit: "%",
            color: "80,220,140",
        },
        {
            label: "Density",
            value: 40,
            target: 40,
            unit: "%",
            color: "180,120,255",
        },
        {
            label: "Brightness",
            value: 50,
            target: 88,
            unit: "%",
            color: "255,180,60",
        },
        {
            label: "Pitch",
            value: 0,
            target: 0,
            unit: "st",
            color: "200,160,80",
        },
    ],
]

function Slide3_ShapeDetail(props: {
    width?: number
    height?: number
    style?: React.CSSProperties
}) {
    const { width = 800, height = 500, style } = props
    const [paramPhase, setParamPhase] = useState(0)
    const [params, setParams] = useState<ParamDef[]>(paramSets[0])
    const animRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const cycleRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    // Animate params toward their targets
    useEffect(() => {
        const currentSet = paramSets[paramPhase]
        setParams(currentSet.map((p) => ({ ...p })))

        let frame = 0
        const totalFrames = 45
        animRef.current = setInterval(() => {
            frame++
            const t = Math.min(frame / totalFrames, 1)
            const ease = 1 - Math.pow(1 - t, 3)
            setParams(
                currentSet.map((p) => ({
                    ...p,
                    value: Math.round(p.value + (p.target - p.value) * ease),
                }))
            )
            if (frame >= totalFrames) {
                if (animRef.current) clearInterval(animRef.current)
            }
        }, 33)

        return () => {
            if (animRef.current) clearInterval(animRef.current)
        }
    }, [paramPhase])

    // Cycle through param sets
    useEffect(() => {
        const run = () => {
            setParamPhase(0)
            setTimeout(() => setParamPhase(1), 2500)
            setTimeout(() => setParamPhase(2), 5000)
            cycleRef.current = setTimeout(run, CYCLE)
        }
        run()
        return () => {
            if (cycleRef.current) clearTimeout(cycleRef.current)
        }
    }, [])

    return (
        <SlideFrame
            glowRGB={GLOW}
            headline="Make literally any sound possible."
            copy="Our  timbre shaping models allow you to generate and shape sounds with incredible control."
            width={width}
            height={height}
            style={style}
        >
            <div
                style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    gap: 12,
                }}
            >
                {/* Parameter panel (left) */}
                <div
                    style={{
                        width: 200,
                        flexShrink: 0,
                        ...glassPanel(GLOW, { radius: 14 }),
                        padding: 14,
                        boxSizing: "border-box",
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                        overflow: "hidden",
                    }}
                >
                    <div
                        style={{
                            fontSize: 12,
                            fontWeight: 700,
                            color: C.text,
                            paddingBottom: 10,
                            borderBottom: "1px solid rgba(255,255,255,0.06)",
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                        }}
                    >
                        <div style={{ color: `rgba(${GLOW},0.8)` }}>
                            {Icons.sliders}
                        </div>
                        Effects
                    </div>

                    {params.map((p, i) => (
                        <div key={p.label}>
                            <div
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    marginBottom: 6,
                                }}
                            >
                                <span
                                    style={{
                                        fontSize: 10,
                                        color: C.textMuted,
                                        fontWeight: 600,
                                        textTransform: "uppercase",
                                        letterSpacing: 0.5,
                                    }}
                                >
                                    {p.label}
                                </span>
                                <motion.span
                                    key={`${p.label}-${p.value}`}
                                    initial={{
                                        scale: 1.2,
                                        color: `rgba(${p.color},1)`,
                                    }}
                                    animate={{ scale: 1, color: C.textSec }}
                                    transition={{ duration: 0.3 }}
                                    style={{
                                        fontSize: 11,
                                        fontWeight: 600,
                                        fontVariantNumeric: "tabular-nums",
                                    }}
                                >
                                    {p.label === "Pitch"
                                        ? p.value > 0
                                            ? `+${p.value}`
                                            : `${p.value}`
                                        : p.value}
                                    {p.unit}
                                </motion.span>
                            </div>
                            {/* Slider track */}
                            <div
                                style={{
                                    height: 6,
                                    borderRadius: 3,
                                    background: "rgba(255,255,255,0.06)",
                                    boxShadow:
                                        "inset 0 1px 2px rgba(0,0,0,0.3)",
                                    position: "relative",
                                    overflow: "hidden",
                                }}
                            >
                                <motion.div
                                    animate={{
                                        width:
                                            p.label === "Pitch"
                                                ? `${50 + (p.value / 12) * 50}%`
                                                : `${p.value}%`,
                                    }}
                                    transition={{ duration: 0.1 }}
                                    style={{
                                        height: "100%",
                                        borderRadius: 3,
                                        background: `linear-gradient(90deg, rgba(${p.color},0.6), rgba(${p.color},0.9))`,
                                        boxShadow: `0 0 8px rgba(${p.color},0.3)`,
                                    }}
                                />
                            </div>
                            {/* Slider knob */}
                            <div style={{ position: "relative", height: 0 }}>
                                <motion.div
                                    animate={{
                                        left:
                                            p.label === "Pitch"
                                                ? `calc(${50 + (p.value / 12) * 50}% - 6px)`
                                                : `calc(${p.value}% - 6px)`,
                                    }}
                                    transition={{ duration: 0.1 }}
                                    style={{
                                        position: "absolute",
                                        top: -9,
                                        width: 12,
                                        height: 12,
                                        borderRadius: "50%",
                                        background: `rgba(${p.color},0.9)`,
                                        border: "2px solid rgba(255,255,255,0.3)",
                                        boxShadow: `0 0 10px rgba(${p.color},0.4), 0 2px 4px rgba(0,0,0,0.4)`,
                                    }}
                                />
                            </div>
                        </div>
                    ))}

                    {/* Active param indicator */}
                    <div
                        style={{
                            marginTop: "auto",
                            padding: "8px 12px",
                            borderRadius: 8,
                            background: `rgba(${GLOW},0.08)`,
                            border: `1px solid rgba(${GLOW},0.15)`,
                            fontSize: 10,
                            color: `rgba(${GLOW},0.9)`,
                            fontWeight: 600,
                            textAlign: "center",
                        }}
                    >
                        {paramPhase === 0
                            ? "Tightening space..."
                            : paramPhase === 1
                              ? "Boosting clarity..."
                              : "Brightening tone..."}
                    </div>
                </div>

                {/* Waveform display (right) */}
                <div
                    style={{
                        flex: 1,
                        ...glassPanel(GLOW, { radius: 14 }),
                        overflow: "hidden",
                        display: "flex",
                        flexDirection: "column",
                    }}
                >
                    {/* Tab bar */}
                    <div
                        style={{
                            display: "flex",
                            gap: 0,
                            borderBottom: "1px solid rgba(255,255,255,0.06)",
                            background: "rgba(6,6,12,0.25)",
                        }}
                    >
                        {["Waveform", "Spectrum", "Stereo"].map((tab, i) => (
                            <div
                                key={tab}
                                style={{
                                    padding: "7px 14px",
                                    fontSize: 10,
                                    fontWeight: 500,
                                    color:
                                        i === 0
                                            ? `rgba(${GLOW},1)`
                                            : C.textMuted,
                                    borderBottom:
                                        i === 0
                                            ? `2px solid rgba(${GLOW},1)`
                                            : "2px solid transparent",
                                    background:
                                        i === 0
                                            ? `rgba(${GLOW},0.06)`
                                            : "transparent",
                                }}
                            >
                                {tab}
                            </div>
                        ))}
                    </div>

                    {/* Waveform area */}
                    <div
                        style={{
                            flex: 1,
                            position: "relative",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            padding: 16,
                        }}
                    >
                        {/* Before/After label */}
                        <div
                            style={{
                                position: "absolute",
                                top: 8,
                                left: 12,
                                display: "flex",
                                gap: 8,
                            }}
                        >
                            <span
                                style={{
                                    fontSize: 9,
                                    fontWeight: 600,
                                    color: "rgba(255,255,255,0.3)",
                                    padding: "2px 8px",
                                    borderRadius: 4,
                                    background: "rgba(255,255,255,0.05)",
                                }}
                            >
                                BEFORE
                            </span>
                            <span
                                style={{
                                    fontSize: 9,
                                    fontWeight: 600,
                                    color: `rgba(${GLOW},0.8)`,
                                    padding: "2px 8px",
                                    borderRadius: 4,
                                    background: `rgba(${GLOW},0.1)`,
                                }}
                            >
                                AFTER
                            </span>
                        </div>

                        {/* Morphing waveform canvas */}
                        <MorphingWaveform
                            phaseIndex={paramPhase}
                            color={`rgba(${GLOW},0.7)`}
                            beforeColor="rgba(255,255,255,0.07)"
                        />

                        {/* Glow pulse when params change */}
                        <motion.div
                            key={paramPhase}
                            initial={{ opacity: 0.6, scale: 1.02 }}
                            animate={{ opacity: 0, scale: 1 }}
                            transition={{ duration: 1 }}
                            style={{
                                position: "absolute",
                                inset: -20,
                                background: `radial-gradient(ellipse at center, rgba(${GLOW},0.15) 0%, transparent 70%)`,
                                pointerEvents: "none",
                            }}
                        />
                    </div>
                </div>
            </div>
        </SlideFrame>
    )
}

addPropertyControls(Slide3_ShapeDetail, {
    width: { type: ControlType.Number, defaultValue: 800, min: 400, max: 1400 },
    height: { type: ControlType.Number, defaultValue: 500, min: 300, max: 800 },
})

export default Slide3_ShapeDetail
