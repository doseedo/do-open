/**
 * Slide 6 — "See every layer."
 * DAW panel always visible. The rotating stem canvas sits inside a
 * "Full Mix" track row, then splits into 4 separate stem tracks.
 * Loops every 10 seconds.
 */
import React, { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { addPropertyControls, ControlType } from "./framer-stub"
import {
    C,
    Icons,
    WaveformSVG,
    NoiseWaveform,
    glassPanel,
    SlideFrame,
    TransportBar,
    Timeline,
} from "./shared"

const stemDefs = [
    {
        name: "Vocals",
        rgb: [236, 72, 153],
        color: "rgba(236,72,153,0.7)",
        colorBg: "rgba(236,72,153,0.06)",
        icon: Icons.mic,
        seed: 501,
    },
    {
        name: "Celeste",
        rgb: [59, 130, 246],
        color: "rgba(59,130,246,0.7)",
        colorBg: "rgba(59,130,246,0.06)",
        icon: Icons.piano,
        seed: 502,
    },
    {
        name: "Strings",
        rgb: [16, 185, 129],
        color: "rgba(16,185,129,0.7)",
        colorBg: "rgba(16,185,129,0.06)",
        icon: Icons.violin,
        seed: 503,
    },
    {
        name: "Winds",
        rgb: [245, 158, 11],
        color: "rgba(245,158,11,0.7)",
        colorBg: "rgba(245,158,11,0.06)",
        icon: Icons.wind,
        seed: 504,
    },
]

const GLOW = "120,60,200"
const CYCLE_MS = 1300
const ROTATE_TIME = CYCLE_MS // one full rotation then split
const TOTAL_CYCLE = 6000

const BAR_W = 2
const BAR_GAP = 1
const BAR_STEP = BAR_W + BAR_GAP

function seededRandom(seed: number) {
    let s = seed
    return () => {
        s = (s * 16807 + 0) % 2147483647
        return (s - 1) / 2147483646
    }
}

function buildWaveData(barCount: number) {
    const n = stemDefs.length
    const stemSeeds = [101, 247, 389, 513]
    const stemAmps: number[][] = []
    for (let s = 0; s < n; s++) {
        const rng = seededRandom(stemSeeds[s])
        const amps: number[] = []
        for (let b = 0; b < barCount; b++) {
            const pos = b / barCount
            let env: number
            switch (s) {
                case 0:
                    env = 0.2 + 0.8 * Math.pow(Math.sin(pos * Math.PI), 1.2)
                    break
                case 1:
                    env =
                        0.1 +
                        0.9 *
                            Math.pow(Math.sin(pos * Math.PI * 3 + 0.5), 2) *
                            0.7
                    break
                case 2:
                    env =
                        0.15 +
                        0.85 *
                            Math.pow(pos, 0.6) *
                            Math.sin(pos * Math.PI * 0.9)
                    break
                case 3:
                    env =
                        0.1 +
                        0.9 *
                            (0.5 +
                                0.5 * Math.cos(pos * Math.PI * 2 - Math.PI)) *
                            0.6
                    break
                default:
                    env = 0.5
            }
            amps.push(rng() * env * (rng() > 0.88 ? 1.4 : 1))
        }
        stemAmps.push(amps)
    }
    const masterAmps: number[] = []
    for (let b = 0; b < barCount; b++) {
        let mx = 0
        for (let s = 0; s < n; s++) {
            if (stemAmps[s][b] > mx) mx = stemAmps[s][b]
        }
        masterAmps.push(mx)
    }
    const peak = Math.max(...masterAmps)
    for (let b = 0; b < barCount; b++) {
        masterAmps[b] /= peak
        for (let s = 0; s < n; s++) stemAmps[s][b] /= peak
    }
    return { masterAmps, stemAmps }
}

// ── Canvas waveform sized to fit a track row ──
function RotatingWaveform({
    width,
    height,
    startTime,
    freeze = false,
}: {
    width: number
    height: number
    startTime: number
    freeze?: boolean
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const animRef = useRef<number>(0)
    const dataRef = useRef<ReturnType<typeof buildWaveData> | null>(null)
    const barCount = Math.floor(width / BAR_STEP)

    if (!dataRef.current || dataRef.current.masterAmps.length !== barCount) {
        dataRef.current = buildWaveData(barCount)
    }

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")!
        const dpr = window.devicePixelRatio || 1
        canvas.width = width * dpr
        canvas.height = height * dpr
        canvas.style.width = `${width}px`
        canvas.style.height = `${height}px`
        ctx.scale(dpr, dpr)

        const { masterAmps, stemAmps } = dataRef.current!
        const centerY = height / 2
        const maxAmp = height * 0.44
        const n = stemDefs.length

        // Smooth amplitude data with a moving average
        const smooth = (amps: number[], radius: number = 3): number[] => {
            const out: number[] = []
            for (let i = 0; i < amps.length; i++) {
                let sum = 0,
                    count = 0
                for (
                    let j = Math.max(0, i - radius);
                    j <= Math.min(amps.length - 1, i + radius);
                    j++
                ) {
                    sum += amps[j]
                    count++
                }
                out.push(sum / count)
            }
            return out
        }

        const barW = 3
        const barGap = 1.5
        const barR = barW / 2 // full round caps

        // Helper: draw rounded bars from amplitude array
        const drawWave = (rawAmps: number[], style: string) => {
            const amps = smooth(rawAmps)
            ctx.fillStyle = style
            for (let b = 0; b < barCount; b++) {
                const x = b * (barW + barGap)
                if (x > width) break
                const h = amps[b] * maxAmp
                if (h < 1) continue
                ctx.beginPath()
                ctx.roundRect(x, centerY - h, barW, h * 2, barR)
                ctx.fill()
            }
        }

        const animate = () => {
            const t = (Date.now() - startTime) / CYCLE_MS

            ctx.clearRect(0, 0, width, height)

            // Grey master outline
            drawWave(masterAmps, "rgba(255,255,255,0.1)")

            if (!freeze) {
                // Color cycle stems
                for (let s = 0; s < n; s++) {
                    const ph = (((t - s / n) % 1) + 1) % 1
                    const alpha = Math.max(0, Math.cos(ph * 2 * Math.PI))
                    if (alpha > 0.01) {
                        const [r, g, b_] = stemDefs[s].rgb
                        drawWave(
                            stemAmps[s],
                            `rgba(${r},${g},${b_},${alpha * 0.85})`
                        )
                    }
                }
            } else {
                // Frozen: purple master fill
                drawWave(masterAmps, "rgba(168,127,255,0.5)")
            }

            animRef.current = requestAnimationFrame(animate)
        }
        animRef.current = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(animRef.current)
    }, [width, height, barCount, startTime, freeze])

    return <canvas ref={canvasRef} style={{ display: "block" }} />
}

// ── Main component ──
function Slide6_StemViz(props: {
    width?: number
    height?: number
    style?: React.CSSProperties
}) {
    const { width = 800, height = 500, style } = props
    const [phase, setPhase] = useState<
        "rotating" | "splitting" | "generating" | "done"
    >("rotating")
    const [revealedStems, setRevealedStems] = useState<number[]>([])
    const cycleRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const startRef = useRef(Date.now())

    const labelW = 100
    const trackH = 52

    useEffect(() => {
        const run = () => {
            setPhase("rotating")
            setRevealedStems([])
            startRef.current = Date.now()

            setTimeout(() => setPhase("splitting"), ROTATE_TIME)
            setTimeout(() => setPhase("generating"), ROTATE_TIME + 700)

            stemDefs.forEach((_, i) => {
                setTimeout(
                    () => {
                        setRevealedStems((prev) => [...prev, i])
                    },
                    ROTATE_TIME + 1400 + i * 400
                )
            })

            setTimeout(
                () => setPhase("done"),
                ROTATE_TIME + 1400 + stemDefs.length * 400 + 200
            )
            cycleRef.current = setTimeout(run, TOTAL_CYCLE)
        }
        run()
        return () => {
            if (cycleRef.current) clearTimeout(cycleRef.current)
        }
    }, [])

    const showMix = phase === "rotating"
    const isSplit =
        phase === "splitting" || phase === "generating" || phase === "done"

    return (
        <SlideFrame
            glowRGB={GLOW}
            headline="Turn your songs back into sessions."
            copy="Our state of the art source separation and reverse FX models allow instant stems with equivalent dry recordings through extracted FX chains, from only a master recording."
            width={width}
            height={height}
            style={style}
        >
            <div
                style={{
                    position: "absolute",
                    inset: 0,
                    ...glassPanel(GLOW, { radius: 14 }),
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                }}
            >
                <TransportBar glowRGB={GLOW} isPlaying={false} time="0:00" />
                <Timeline duration={30} labelW={labelW} />

                <div
                    style={{
                        flex: 1,
                        position: "relative",
                        overflow: "hidden",
                    }}
                >
                    {/* Row backgrounds */}
                    {[0, 1, 2, 3].map((i) => (
                        <div
                            key={i}
                            style={{
                                position: "absolute",
                                top: i * trackH,
                                left: 0,
                                right: 0,
                                height: trackH,
                                borderBottom:
                                    "1px solid rgba(255,255,255,0.04)",
                            }}
                        >
                            <div
                                style={{
                                    position: "absolute",
                                    left: 0,
                                    top: 0,
                                    bottom: 0,
                                    width: labelW,
                                    background: "rgba(8,8,14,0.2)",
                                    borderRight:
                                        "1px solid rgba(255,255,255,0.06)",
                                }}
                            />
                        </div>
                    ))}

                    {/* Full Mix track with rotating canvas — spans rows 1-2 */}
                    <AnimatePresence>
                        {showMix && (
                            <>
                                {/* Label */}
                                <motion.div
                                    key="mix-label"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -10 }}
                                    transition={{
                                        duration: 0.5,
                                        ease: [0.2, 0.8, 0.2, 1],
                                    }}
                                    style={{
                                        position: "absolute",
                                        top: trackH * 0.5,
                                        left: 0,
                                        width: labelW,
                                        height: trackH * 3,
                                        display: "flex",
                                        alignItems: "center",
                                        padding: "0 6px",
                                        gap: 6,
                                        background: "rgba(8,8,14,0.4)",
                                        backdropFilter: "blur(20px)",
                                        borderRight:
                                            "1px solid rgba(255,255,255,0.06)",
                                        zIndex: 2,
                                    }}
                                >
                                    <div
                                        style={{
                                            color: "rgba(168,127,255,0.7)",
                                            opacity: 0.7,
                                        }}
                                    >
                                        {Icons.mic}
                                    </div>
                                    <span
                                        style={{
                                            fontSize: 10,
                                            color: C.textSec,
                                            fontWeight: 500,
                                        }}
                                    >
                                        Full Mix
                                    </span>
                                </motion.div>

                                {/* Canvas clip — drops in from above */}
                                <motion.div
                                    key="mix-clip"
                                    initial={{
                                        opacity: 0,
                                        y: -30,
                                        scaleY: 0.7,
                                    }}
                                    animate={{ opacity: 1, y: 0, scaleY: 1 }}
                                    exit={{
                                        opacity: 0,
                                        scaleY: 0.3,
                                        filter: "blur(8px)",
                                    }}
                                    transition={{
                                        duration: 0.5,
                                        ease: [0.2, 0.8, 0.2, 1],
                                    }}
                                    style={{
                                        position: "absolute",
                                        top: trackH * 0.5 + 3,
                                        left: labelW + 8,
                                        right: 8,
                                        height: trackH * 3 - 6,
                                        borderRadius: 8,
                                        overflow: "hidden",
                                        background:
                                            "linear-gradient(180deg, rgba(168,127,255,0.08), rgba(168,127,255,0.02))",
                                        borderLeft:
                                            "2px solid rgba(168,127,255,0.7)",
                                        borderRight:
                                            "1px solid rgba(168,127,255,0.12)",
                                        borderTop:
                                            "1px solid rgba(168,127,255,0.15)",
                                        borderBottom:
                                            "1px solid rgba(168,127,255,0.05)",
                                        boxShadow:
                                            "0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08)",
                                    }}
                                >
                                    <RotatingWaveform
                                        width={550}
                                        height={trackH * 3 - 10}
                                        startTime={startRef.current}
                                    />
                                </motion.div>
                            </>
                        )}
                    </AnimatePresence>

                    {/* Status badge — stays visible, text swaps */}
                    <AnimatePresence>
                        {(phase === "rotating" || phase === "splitting") && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                style={{
                                    position: "absolute",
                                    top: "50%",
                                    left: "50%",
                                    transform: "translate(-50%, -50%)",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 8,
                                    padding: "8px 16px",
                                    borderRadius: 20,
                                    background: `rgba(${GLOW},0.15)`,
                                    backdropFilter: "blur(20px)",
                                    border: `1px solid rgba(${GLOW},0.3)`,
                                    zIndex: 10,
                                }}
                            >
                                <div style={{ color: `rgba(${GLOW},0.9)` }}>
                                    {phase === "rotating"
                                        ? Icons.wand
                                        : Icons.split}
                                </div>
                                <span
                                    style={{
                                        fontSize: 11,
                                        color: "#fff",
                                        fontWeight: 600,
                                    }}
                                >
                                    {phase === "rotating"
                                        ? "Analyzing stems..."
                                        : "Separating stems..."}
                                </span>
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{
                                        duration: 1,
                                        repeat: Infinity,
                                        ease: "linear",
                                    }}
                                    style={{
                                        width: 14,
                                        height: 14,
                                        borderRadius: "50%",
                                        border: `2px solid rgba(${GLOW},0.2)`,
                                        borderTopColor: `rgba(${GLOW},0.8)`,
                                    }}
                                />
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Individual stem tracks */}
                    {isSplit &&
                        stemDefs.map((stem, i) => {
                            const isRevealed = revealedStems.includes(i)
                            return (
                                <motion.div
                                    key={stem.name}
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{
                                        duration: 0.3,
                                        delay: i * 0.06,
                                    }}
                                >
                                    {/* Label */}
                                    <div
                                        style={{
                                            position: "absolute",
                                            top: i * trackH,
                                            left: 0,
                                            width: labelW,
                                            height: trackH,
                                            display: "flex",
                                            alignItems: "center",
                                            padding: "0 6px",
                                            gap: 6,
                                            background: "rgba(8,8,14,0.4)",
                                            backdropFilter: "blur(20px)",
                                            borderRight:
                                                "1px solid rgba(255,255,255,0.06)",
                                            borderBottom:
                                                "1px solid rgba(255,255,255,0.04)",
                                            zIndex: 2,
                                        }}
                                    >
                                        <div
                                            style={{
                                                color: stem.color,
                                                opacity: 0.7,
                                            }}
                                        >
                                            {stem.icon}
                                        </div>
                                        <span
                                            style={{
                                                fontSize: 10,
                                                color: C.textSec,
                                                fontWeight: 500,
                                            }}
                                        >
                                            {stem.name}
                                        </span>
                                    </div>

                                    {/* Clip */}
                                    <div
                                        style={{
                                            position: "absolute",
                                            top: i * trackH + 3,
                                            left: labelW + 8,
                                            right: 8,
                                            height: trackH - 6,
                                            overflow: "hidden",
                                            borderRadius: 8,
                                        }}
                                    >
                                        {/* Noise always renders underneath */}
                                        <div
                                            style={{
                                                position: "absolute",
                                                inset: 0,
                                                borderRadius: 8,
                                                background: `rgba(${GLOW},0.04)`,
                                                border: `1px solid rgba(${GLOW},0.12)`,
                                                opacity: isRevealed ? 0 : 1,
                                                transition: "opacity 0.4s ease",
                                            }}
                                        >
                                            <NoiseWaveform
                                                width={500}
                                                height={trackH - 10}
                                                color={stem.color}
                                                settling={isRevealed}
                                            />
                                        </div>

                                        {/* Revealed waveform fades in on top */}
                                        <div
                                            style={{
                                                position: "absolute",
                                                inset: 0,
                                                opacity: isRevealed ? 1 : 0,
                                                transition: "opacity 0.5s ease",
                                                background: `linear-gradient(180deg, ${stem.colorBg}, ${stem.colorBg.replace("0.06", "0.02")})`,
                                                backdropFilter:
                                                    "blur(14px) saturate(150%)",
                                                borderLeft: `2px solid ${stem.color}`,
                                                borderRight: `1px solid ${stem.color.replace("0.7", "0.12")}`,
                                                borderTop: `1px solid ${stem.color.replace("0.7", "0.15")}`,
                                                borderBottom: `1px solid ${stem.color.replace("0.7", "0.05")}`,
                                                borderRadius: 8,
                                                boxShadow: `0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 20px ${stem.color.replace("0.7", "0.06")}`,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    position: "absolute",
                                                    left: 6,
                                                    top: "50%",
                                                    transform:
                                                        "translateY(-50%)",
                                                }}
                                            >
                                                <WaveformSVG
                                                    color={stem.color}
                                                    seed={stem.seed}
                                                    width={500}
                                                    height={trackH - 20}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )
                        })}
                </div>
            </div>
        </SlideFrame>
    )
}

addPropertyControls(Slide6_StemViz, {
    width: { type: ControlType.Number, defaultValue: 800, min: 400, max: 1400 },
    height: { type: ControlType.Number, defaultValue: 500, min: 300, max: 800 },
})

export default Slide6_StemViz
