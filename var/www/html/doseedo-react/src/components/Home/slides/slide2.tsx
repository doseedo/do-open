/**
 * Slide 7 — "Track-aware generation"
 *
 * KEY FIX: Zero requestAnimationFrame → setState calls.
 * The playhead and spinner are updated by direct DOM mutation via refs.
 * Only setTimeout-driven phase changes use setState (slow, infrequent).
 * This prevents 60fps React re-renders which cause Framer's compositor to black out.
 */
import React, { useState, useEffect, useRef } from "react"
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

const GLOW = "100,160,255"
const CYCLE = 10000
const SEL_START = 0.35
const SEL_END = 0.78
const LW = 100
const TH = 52
const PAD = 8

function seededRng(seed: number) {
    let s = seed
    return () => {
        s = (s * 16807) % 2147483647
        return (s - 1) / 2147483646
    }
}

function LocalWaveform({
    color,
    seed,
    w,
    h,
}: {
    color: string
    seed: number
    w: number
    h: number
}) {
    const rng = seededRng(seed)
    const mid = h / 2
    let d = ""
    for (let x = 0; x < w; x += 3) {
        const env = 0.4 + 0.6 * Math.sin((x / w) * Math.PI)
        const amp = rng() * env * h * 0.42 * (rng() > 0.87 ? 1.3 : 1)
        d += `M${x},${mid - amp}V${mid + amp}`
    }
    return (
        <svg width={w} height={h} style={{ display: "block" }}>
            <path
                d={d}
                stroke={color}
                strokeWidth={2}
                fill="none"
                strokeLinecap="round"
            />
        </svg>
    )
}

function NoiseCanvas({ w, h, color }: { w: number; h: number; color: string }) {
    const ref = useRef<HTMLCanvasElement>(null)
    const raf = useRef(0)
    const amps = useRef<number[]>([])
    useEffect(() => {
        const canvas = ref.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")!
        const dpr = window.devicePixelRatio || 1
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.scale(dpr, dpr)
        const count = Math.floor(w / 4)
        amps.current = new Array(count).fill(0)
        const mid = h / 2
        const draw = () => {
            ctx.clearRect(0, 0, w, h)
            ctx.strokeStyle = color
            ctx.lineWidth = 2
            for (let i = 0; i < count; i++) {
                const x = (i / count) * w
                amps.current[i] +=
                    ((Math.random() - 0.5) * h * 0.6 - amps.current[i]) * 0.08
                ctx.beginPath()
                ctx.moveTo(x, mid - amps.current[i])
                ctx.lineTo(x, mid + amps.current[i])
                ctx.stroke()
            }
            raf.current = requestAnimationFrame(draw)
        }
        draw()
        return () => cancelAnimationFrame(raf.current)
    }, [w, h, color])
    return (
        <canvas ref={ref} style={{ width: w, height: h, display: "block" }} />
    )
}

const TRACKS = [
    {
        name: "Piano",
        color: "rgba(72,202,228,0.8)",
        bg: "rgba(72,202,228,0.07)",
        seed: 701,
        genSeed: 801,
    },
    {
        name: "Bass",
        color: "rgba(255,100,100,0.8)",
        bg: "rgba(255,100,100,0.07)",
        seed: 702,
        genSeed: 802,
    },
    {
        name: "Drums",
        color: "rgba(255,165,0,0.8)",
        bg: "rgba(255,165,0,0.07)",
        seed: 703,
        genSeed: 803,
    },
    {
        name: "Trumpet",
        color: "rgba(245,158,11,0.8)",
        bg: "rgba(245,158,11,0.07)",
        seed: 704,
        genSeed: 804,
    },
]

type Phase =
    | "idle"
    | "selecting"
    | "splitting"
    | "generating"
    | "resolving"
    | "playing"
    | "fadeout"

function Slide7_Generate(props: {
    width?: number
    height?: number
    style?: React.CSSProperties
}) {
    const { width = 800, height = 500, style } = props

    // Phase state — only changes ~7 times per cycle via setTimeout
    const [phase, setPhase] = useState<Phase>("idle")
    const [resolved, setResolved] = useState<number[]>([])
    const [selPrg, setSelPrg] = useState(0) // 0..1 selection box progress

    // DOM refs for rAF-driven elements (no setState)
    const playheadLineRef = useRef<HTMLDivElement>(null)
    const playheadCapRef = useRef<HTMLDivElement>(null)
    const timeDisplayRef = useRef<HTMLDivElement>(null)
    const spinnerRef = useRef<HTMLDivElement>(null)

    const rafRef = useRef(0)
    const selRafRef = useRef(0)
    const timers = useRef<ReturnType<typeof setTimeout>[]>([])
    const cycleTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
    const playStart = useRef(0)
    const selStart = useRef(0)
    const spinAngle = useRef(0)
    const lastFrame = useRef(0)

    const trackAreaW = width - LW - PAD * 2
    const selStartPx = LW + PAD + SEL_START * trackAreaW
    const selWidthPx = (SEL_END - SEL_START) * trackAreaW
    const transportH = 32
    const timelineH = 24
    const headerH = transportH + timelineH
    const panelW = width - 80
    const panelH = height - 160
    const beforeW = SEL_START * trackAreaW - 2
    const genW = selWidthPx - 2
    const afterLeft = selStartPx + selWidthPx + 2
    const afterW = Math.max(0, panelW - afterLeft - PAD)

    // rAF loop — only mutates DOM, never calls setState
    useEffect(() => {
        const tick = (now: number) => {
            const dt = now - lastFrame.current
            lastFrame.current = now

            // Playhead
            const elapsed = (now - playStart.current) / 1000
            const ph = (elapsed % 30) / 30
            const phPx = LW + PAD + ph * trackAreaW

            if (playheadLineRef.current) {
                playheadLineRef.current.style.left = `${phPx}px`
            }
            if (playheadCapRef.current) {
                playheadCapRef.current.style.left = `${phPx - 4}px`
            }
            if (timeDisplayRef.current) {
                timeDisplayRef.current.textContent = `0:${String(Math.floor(ph * 30)).padStart(2, "0")}`
            }

            // Spinner
            if (spinnerRef.current) {
                spinAngle.current = (spinAngle.current + dt * 0.36) % 360
                spinnerRef.current.style.transform = `rotate(${spinAngle.current}deg)`
            }

            rafRef.current = requestAnimationFrame(tick)
        }
        lastFrame.current = performance.now()
        rafRef.current = requestAnimationFrame(tick)
        return () => cancelAnimationFrame(rafRef.current)
    }, [trackAreaW])

    // Selection box animation — uses setState but only runs for 800ms
    const runSelAnim = () => {
        selStart.current = Date.now()
        const step = () => {
            const p = Math.min(1, (Date.now() - selStart.current) / 800)
            setSelPrg(1 - Math.pow(1 - p, 3))
            if (p < 1) selRafRef.current = requestAnimationFrame(step)
        }
        selRafRef.current = requestAnimationFrame(step)
    }

    // Phase cycle
    useEffect(() => {
        const addTimer = (ms: number, fn: () => void) => {
            const id = setTimeout(fn, ms)
            timers.current.push(id)
            return id
        }

        const run = () => {
            timers.current.forEach(clearTimeout)
            timers.current = []
            cancelAnimationFrame(selRafRef.current)

            setPhase("idle")
            setResolved([])
            setSelPrg(0)

            playStart.current = Date.now() - (SEL_START * 30000 - 6250)

            addTimer(1500, () => {
                setPhase("selecting")
                runSelAnim()
            })
            addTimer(2500, () => setPhase("splitting"))
            addTimer(3000, () => setPhase("generating"))
            addTimer(4500, () => setPhase("resolving"))
            TRACKS.forEach((_, i) =>
                addTimer(4500 + i * 300, () => setResolved((p) => [...p, i]))
            )
            addTimer(6500, () => setPhase("playing"))
            addTimer(CYCLE - 1000, () => setPhase("fadeout"))
            cycleTimer.current = setTimeout(run, CYCLE)
        }

        run()
        return () => {
            timers.current.forEach(clearTimeout)
            if (cycleTimer.current) clearTimeout(cycleTimer.current)
            cancelAnimationFrame(rafRef.current)
            cancelAnimationFrame(selRafRef.current)
        }
    }, [])

    const isSplit = [
        "splitting",
        "generating",
        "resolving",
        "playing",
        "fadeout",
    ].includes(phase)
    const showNoise = [
        "generating",
        "resolving",
        "playing",
        "fadeout",
    ].includes(phase)
    const showMarquee = [
        "selecting",
        "splitting",
        "generating",
        "resolving",
    ].includes(phase)
    const showBadge = phase === "generating" || phase === "splitting"
    const isSelecting = phase === "selecting"

    const clip = (color: string, bg: string): React.CSSProperties => ({
        borderRadius: 8,
        background: bg,
        borderLeft: `2px solid ${color}`,
        boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
    })

    const ticks = [0, 5, 10, 15, 20, 25, 30]

    return (
        <div
            style={{
                width,
                height,
                position: "relative",
                background: "#050508",
                fontFamily:
                    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif",
                userSelect: "none",
                ...style,
            }}
        >
            {/* Glow */}
            <div
                style={{
                    position: "absolute",
                    top: -40,
                    left: -40,
                    right: -40,
                    bottom: -40,
                    background: `radial-gradient(ellipse at 50% 40%, rgba(${GLOW},0.16) 0%, rgba(${GLOW},0.05) 45%, transparent 70%)`,
                    pointerEvents: "none",
                }}
            />

            {/* Panel */}
            <div
                style={{
                    position: "absolute",
                    top: 40,
                    left: 40,
                    width: panelW,
                    height: panelH,
                    background: "rgba(8,8,14,0.92)",
                    borderRadius: 14,
                    border: `1px solid rgba(${GLOW},0.15)`,
                    opacity: phase === "fadeout" ? 0 : 1,
                    transition: phase === "fadeout" ? "opacity 0.8s" : "none",
                }}
            >
                {/* Transport */}
                <div
                    style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: transportH,
                        background: "rgba(5,5,10,0.85)",
                        borderBottom: "1px solid rgba(255,255,255,0.07)",
                        borderRadius: "14px 14px 0 0",
                        display: "flex",
                        alignItems: "center",
                        padding: "0 10px",
                        gap: 6,
                    }}
                >
                    <div
                        style={{
                            width: 24,
                            height: 24,
                            borderRadius: 6,
                            border: `1px solid rgba(${GLOW},0.45)`,
                            background: `rgba(${GLOW},0.18)`,
                            color: "#fff",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <svg
                            width={10}
                            height={12}
                            viewBox="0 0 10 12"
                            fill="currentColor"
                        >
                            <rect x="0" y="0" width="3" height="12" rx="1" />
                            <rect x="7" y="0" width="3" height="12" rx="1" />
                        </svg>
                    </div>
                    <div
                        style={{
                            width: 24,
                            height: 24,
                            borderRadius: 6,
                            border: "1px solid rgba(255,255,255,0.1)",
                            background: "rgba(255,255,255,0.05)",
                            color: "#aaa",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <svg
                            width={10}
                            height={10}
                            viewBox="0 0 10 10"
                            fill="currentColor"
                        >
                            <rect x="0" y="0" width="10" height="10" rx="1" />
                        </svg>
                    </div>
                    {/* Time display — mutated directly by rAF */}
                    <div
                        ref={timeDisplayRef}
                        style={{
                            color: "#fff",
                            fontSize: 11,
                            fontWeight: 600,
                            minWidth: 40,
                            padding: "0 6px",
                            background: "rgba(5,5,10,0.7)",
                            border: "1px solid rgba(255,255,255,0.07)",
                            borderRadius: 6,
                            height: 24,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontVariantNumeric: "tabular-nums",
                        }}
                    >
                        0:00
                    </div>
                    <div style={{ flex: 1 }} />
                    <div
                        style={{
                            padding: "3px 8px",
                            borderRadius: 6,
                            fontSize: 10,
                            background: "rgba(255,255,255,0.04)",
                            border: "1px solid rgba(255,255,255,0.07)",
                            color: "#666",
                        }}
                    >
                        120 BPM
                    </div>
                </div>

                {/* Timeline */}
                <div
                    style={{
                        position: "absolute",
                        top: transportH,
                        left: 0,
                        right: 0,
                        height: timelineH,
                        background: "rgba(5,5,10,0.55)",
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                        display: "flex",
                    }}
                >
                    <div
                        style={{
                            width: LW,
                            minWidth: LW,
                            background: "rgba(5,5,10,0.4)",
                            borderRight: "1px solid rgba(255,255,255,0.05)",
                            display: "flex",
                            alignItems: "center",
                            paddingLeft: 10,
                        }}
                    >
                        <span style={{ fontSize: 9, color: "#444" }}>
                            + Add Track
                        </span>
                    </div>
                    <div style={{ flex: 1, position: "relative" }}>
                        {ticks.map((tk) => (
                            <div
                                key={tk}
                                style={{
                                    position: "absolute",
                                    left: `${(tk / 30) * 100}%`,
                                    top: 0,
                                    height: "100%",
                                }}
                            >
                                <span
                                    style={{
                                        position: "absolute",
                                        top: 4,
                                        left: 3,
                                        color: "#444",
                                        fontSize: 8,
                                        whiteSpace: "nowrap",
                                    }}
                                >
                                    {tk}s
                                </span>
                                <div
                                    style={{
                                        position: "absolute",
                                        bottom: 0,
                                        left: 0,
                                        width: 1,
                                        height: 8,
                                        background: "#2a2a2a",
                                    }}
                                />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Tracks */}
                {TRACKS.map((track, i) => {
                    const rowTop = headerH + i * TH
                    return (
                        <div key={track.name}>
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop,
                                    left: 0,
                                    right: 0,
                                    height: TH,
                                    borderBottom:
                                        "1px solid rgba(255,255,255,0.035)",
                                }}
                            />
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop,
                                    left: 0,
                                    width: LW,
                                    height: TH,
                                    background: "rgba(5,5,10,0.55)",
                                    borderRight:
                                        "1px solid rgba(255,255,255,0.05)",
                                    display: "flex",
                                    alignItems: "center",
                                    padding: "0 8px",
                                    gap: 6,
                                    zIndex: 2,
                                }}
                            >
                                <div
                                    style={{
                                        width: 7,
                                        height: 7,
                                        borderRadius: "50%",
                                        background: track.color,
                                        flexShrink: 0,
                                    }}
                                />
                                <span
                                    style={{
                                        fontSize: 10,
                                        color: "#bbb",
                                        fontWeight: 500,
                                    }}
                                >
                                    {track.name}
                                </span>
                            </div>

                            {/* Full clip */}
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop + 3,
                                    left: LW + PAD,
                                    right: PAD,
                                    height: TH - 6,
                                    opacity: isSplit
                                        ? 0
                                        : isSelecting
                                          ? 0.3
                                          : 1,
                                    transition: "opacity 0.3s",
                                    ...clip(track.color, track.bg),
                                }}
                            >
                                <div
                                    style={{
                                        position: "absolute",
                                        left: 6,
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                    }}
                                >
                                    <LocalWaveform
                                        color={track.color}
                                        seed={track.seed}
                                        w={trackAreaW - 10}
                                        h={TH - 20}
                                    />
                                </div>
                            </div>

                            {/* Before */}
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop + 3,
                                    left: LW + PAD,
                                    width: beforeW,
                                    height: TH - 6,
                                    opacity: isSplit ? 0.35 : 0,
                                    transition: "opacity 0.3s",
                                    ...clip(track.color, track.bg),
                                }}
                            >
                                <div
                                    style={{
                                        position: "absolute",
                                        left: 6,
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                    }}
                                >
                                    <LocalWaveform
                                        color={track.color}
                                        seed={track.seed}
                                        w={Math.max(10, beforeW - 10)}
                                        h={TH - 20}
                                    />
                                </div>
                            </div>

                            {/* Noise */}
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop + 3,
                                    left: selStartPx,
                                    width: genW,
                                    height: TH - 6,
                                    opacity:
                                        isSplit &&
                                        showNoise &&
                                        !resolved.includes(i)
                                            ? 1
                                            : 0,
                                    transition: "opacity 0.4s",
                                    borderRadius: 8,
                                    background: `rgba(${GLOW},0.04)`,
                                    border: `1px solid rgba(${GLOW},0.15)`,
                                }}
                            >
                                <NoiseCanvas
                                    w={genW}
                                    h={TH - 10}
                                    color={track.color}
                                />
                            </div>

                            {/* Resolved waveform */}
                            <div
                                style={{
                                    position: "absolute",
                                    top: rowTop + 3,
                                    left: selStartPx,
                                    width: genW,
                                    height: TH - 6,
                                    opacity: resolved.includes(i) ? 1 : 0,
                                    transition: "opacity 0.5s",
                                    ...clip(track.color, track.bg),
                                }}
                            >
                                <div
                                    style={{
                                        position: "absolute",
                                        left: 6,
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                    }}
                                >
                                    <LocalWaveform
                                        color={track.color}
                                        seed={track.genSeed}
                                        w={Math.max(10, genW - 10)}
                                        h={TH - 20}
                                    />
                                </div>
                            </div>

                            {/* After */}
                            {afterW > 4 && (
                                <div
                                    style={{
                                        position: "absolute",
                                        top: rowTop + 3,
                                        left: afterLeft,
                                        width: afterW,
                                        height: TH - 6,
                                        opacity: isSplit ? 0.35 : 0,
                                        transition: "opacity 0.3s",
                                        ...clip(track.color, track.bg),
                                    }}
                                >
                                    <div
                                        style={{
                                            position: "absolute",
                                            left: 6,
                                            top: "50%",
                                            transform: "translateY(-50%)",
                                        }}
                                    >
                                        <LocalWaveform
                                            color={track.color}
                                            seed={track.seed + 500}
                                            w={Math.max(10, afterW - 10)}
                                            h={TH - 20}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )
                })}

                {/* Selection box */}
                <div
                    style={{
                        position: "absolute",
                        top: headerH,
                        left: selStartPx,
                        width: isSplit ? selWidthPx : selWidthPx * selPrg,
                        height: TH * 4,
                        opacity: showMarquee ? 1 : 0,
                        transition: "opacity 0.3s",
                        pointerEvents: "none",
                        zIndex: 10,
                        background: `rgba(${GLOW},0.1)`,
                        border: `1.5px solid rgba(${GLOW},0.6)`,
                        borderRadius: 4,
                    }}
                />

                {/* Jazz label */}
                <div
                    style={{
                        position: "absolute",
                        top: headerH + TH * 4 + 6,
                        left: selStartPx + selWidthPx / 2,
                        transform: "translateX(-50%)",
                        padding: "3px 12px",
                        borderRadius: 10,
                        background: `rgba(${GLOW},0.18)`,
                        border: `1px solid rgba(${GLOW},0.4)`,
                        whiteSpace: "nowrap",
                        opacity: selPrg > 0.5 && showMarquee ? 1 : 0,
                        transition: "opacity 0.3s",
                        pointerEvents: "none",
                        zIndex: 10,
                    }}
                >
                    <span
                        style={{
                            fontSize: 10,
                            fontWeight: 600,
                            color: `rgba(${GLOW},0.9)`,
                            letterSpacing: 0.5,
                        }}
                    >
                        Jazz
                    </span>
                </div>

                {/* Badge — always in DOM, opacity only */}
                <div
                    style={{
                        position: "absolute",
                        top: headerH + TH * 2 - 18,
                        left: "50%",
                        transform: "translateX(-50%)",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 16px",
                        borderRadius: 20,
                        background: `rgba(${GLOW},0.22)`,
                        border: `1px solid rgba(${GLOW},0.35)`,
                        zIndex: 30,
                        whiteSpace: "nowrap",
                        opacity: showBadge ? 1 : 0,
                        transition: "opacity 0.3s",
                        pointerEvents: "none",
                    }}
                >
                    <svg
                        width={14}
                        height={14}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke={`rgba(${GLOW},0.9)`}
                        strokeWidth={2}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8l1.4 1.4M17.8 6.2l1.4-1.4M12.2 11.8l-1.4 1.4M3 21l9-9" />
                    </svg>
                    <span
                        style={{ fontSize: 11, color: "#fff", fontWeight: 600 }}
                    >
                        Generating...
                    </span>
                    {/* Spinner — rotated by rAF DOM mutation, no CSS animation */}
                    <div
                        ref={spinnerRef}
                        style={{
                            width: 13,
                            height: 13,
                            borderRadius: "50%",
                            border: `2px solid rgba(${GLOW},0.2)`,
                            borderTopColor: `rgba(${GLOW},0.9)`,
                        }}
                    />
                </div>

                {/* Playhead — moved by rAF DOM mutation */}
                <div
                    ref={playheadLineRef}
                    style={{
                        position: "absolute",
                        top: headerH,
                        left: LW + PAD,
                        width: 1.5,
                        height: TH * 4,
                        background: "rgba(255,255,255,0.85)",
                        zIndex: 20,
                        pointerEvents: "none",
                    }}
                />
                <div
                    ref={playheadCapRef}
                    style={{
                        position: "absolute",
                        top: headerH - 7,
                        left: LW + PAD - 4,
                        width: 0,
                        height: 0,
                        borderLeft: "4px solid transparent",
                        borderRight: "4px solid transparent",
                        borderTop: "7px solid rgba(255,255,255,0.85)",
                        zIndex: 21,
                        pointerEvents: "none",
                    }}
                />
            </div>

            {/* Text overlay removed — slide2 used to inline its own bottom-
                aligned headline + copy block (it bypasses SlideFrame). The
                text now lives in Home.js SLIDE_LABELS so it doesn't slide
                along with the canvas. The "Track-aware generation" copy
                is at index 1 there. */}
        </div>
    )
}

addPropertyControls(Slide7_Generate, {
    width: { type: ControlType.Number, defaultValue: 800, min: 400, max: 1400 },
    height: { type: ControlType.Number, defaultValue: 500, min: 300, max: 800 },
})

export default Slide7_Generate
