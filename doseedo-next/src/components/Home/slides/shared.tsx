import React, { useEffect, useRef } from "react"

// ── Colors ──
export const C = {
    bgDarkest: "#0a0a0a",
    bgDark: "#1a1a1a",
    bgMid: "#1a1a2e",
    bgLight: "#2a2a2a",
    bgLighter: "#333",
    border: "rgba(255,255,255,0.08)",
    borderLight: "#333",
    borderSubtle: "rgb(40,40,40)",
    text: "#fff",
    textSec: "#ccc",
    textMuted: "#888",
    textDim: "#555",
    blue: "#667eea",
    purple: "#8b5cf6",
    purpleDark: "#764ba2",
    green: "#4CAF50",
} as const

// ── Icons ──
const sz = {
    width: 14,
    height: 14,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
}
export const Icons = {
    play: (
        <svg {...sz}>
            <polygon
                points="5 3 19 12 5 21"
                fill="currentColor"
                stroke="none"
            />
        </svg>
    ),
    pause: (
        <svg {...sz}>
            <rect
                x="6"
                y="4"
                width="4"
                height="16"
                fill="currentColor"
                stroke="none"
            />
            <rect
                x="14"
                y="4"
                width="4"
                height="16"
                fill="currentColor"
                stroke="none"
            />
        </svg>
    ),
    stop: (
        <svg {...sz}>
            <rect
                x="6"
                y="6"
                width="12"
                height="12"
                rx="1"
                fill="currentColor"
                stroke="none"
            />
        </svg>
    ),
    mic: (
        <svg {...sz}>
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
    ),
    violin: (
        <svg {...sz}>
            <path d="M12 2v6M9 8h6M7 12c0 2.8 2.2 5 5 5s5-2.2 5-5-2.2-5-5-5-5 2.2-5 5z" />
            <path d="M12 17v5" />
        </svg>
    ),
    piano: (
        <svg {...sz}>
            <rect x="2" y="4" width="20" height="16" rx="2" />
            <line x1="7" y1="4" x2="7" y2="14" />
            <line x1="12" y1="4" x2="12" y2="14" />
            <line x1="17" y1="4" x2="17" y2="14" />
            <line x1="2" y1="14" x2="22" y2="14" />
        </svg>
    ),
    guitar: (
        <svg {...sz}>
            <path d="M18 2l4 4M17.5 6.5l-3 3M11 13c-2 2-5.5 2-5.5 2s0-3.5 2-5.5l7-7 4 4-7 7z" />
            <path d="M9 15l-2 2" />
        </svg>
    ),
    bass: (
        <svg {...sz}>
            <path d="M19 3l2 2M16 6l-2 2M11 11c-2.5 2.5-6 2-6 2s-.5-3.5 2-6l7-7 3 3-6 6z" />
            <circle cx="7" cy="17" r="3" />
        </svg>
    ),
    trumpet: (
        <svg {...sz}>
            <path d="M3 12h6l2-2h4l2 2h4M7 12v5a2 2 0 0 0 4 0v-5M13 10V7a2 2 0 0 1 4 0v3" />
        </svg>
    ),
    wind: (
        <svg {...sz}>
            <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
        </svg>
    ),
    drums: (
        <svg {...sz}>
            <ellipse cx="12" cy="10" rx="9" ry="5" />
            <path d="M3 10v4c0 2.8 4 5 9 5s9-2.2 9-5v-4" />
            <line x1="7" y1="3" x2="10" y2="8" />
            <line x1="17" y1="3" x2="14" y2="8" />
        </svg>
    ),
    synth: (
        <svg {...sz}>
            <rect x="2" y="6" width="20" height="12" rx="2" />
            <line x1="6" y1="10" x2="6" y2="14" />
            <line x1="10" y1="10" x2="10" y2="14" />
            <line x1="14" y1="10" x2="14" y2="14" />
            <line x1="18" y1="10" x2="18" y2="14" />
        </svg>
    ),
    upload: (
        <svg {...sz}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
    ),
    download: (
        <svg {...sz}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
    ),
    film: (
        <svg {...sz}>
            <rect x="2" y="2" width="20" height="20" rx="2" />
            <line x1="7" y1="2" x2="7" y2="22" />
            <line x1="17" y1="2" x2="17" y2="22" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <line x1="2" y1="7" x2="7" y2="7" />
            <line x1="2" y1="17" x2="7" y2="17" />
            <line x1="17" y1="7" x2="22" y2="7" />
            <line x1="17" y1="17" x2="22" y2="17" />
        </svg>
    ),
    sliders: (
        <svg {...sz}>
            <line x1="4" y1="21" x2="4" y2="14" />
            <line x1="4" y1="10" x2="4" y2="3" />
            <line x1="12" y1="21" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12" y2="3" />
            <line x1="20" y1="21" x2="20" y2="16" />
            <line x1="20" y1="12" x2="20" y2="3" />
            <line x1="1" y1="14" x2="7" y2="14" />
            <line x1="9" y1="8" x2="15" y2="8" />
            <line x1="17" y1="16" x2="23" y2="16" />
        </svg>
    ),
    check: (
        <svg {...sz}>
            <polyline points="20 6 9 17 4 12" />
        </svg>
    ),
    split: (
        <svg {...sz}>
            <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
        </svg>
    ),
    wand: (
        <svg {...sz}>
            <path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8l1.4 1.4M17.8 6.2l1.4-1.4M12.2 11.8l-1.4 1.4M3 21l9-9" />
        </svg>
    ),
}

// ── Instrument Data ──
export const instrumentPool = [
    {
        name: "Vocals",
        icon: Icons.mic,
        color: "rgba(168,127,255,0.7)",
        colorBg: "rgba(168,127,255,0.06)",
    },
    {
        name: "Strings",
        icon: Icons.violin,
        color: "rgba(102,126,234,0.7)",
        colorBg: "rgba(102,126,234,0.06)",
    },
    {
        name: "Piano",
        icon: Icons.piano,
        color: "rgba(72,202,228,0.7)",
        colorBg: "rgba(72,202,228,0.06)",
    },
    {
        name: "Winds",
        icon: Icons.wind,
        color: "rgba(100,220,150,0.7)",
        colorBg: "rgba(100,220,150,0.06)",
    },
    {
        name: "Drums",
        icon: Icons.drums,
        color: "rgba(255,165,0,0.7)",
        colorBg: "rgba(255,165,0,0.06)",
    },
    {
        name: "Bass",
        icon: Icons.bass,
        color: "rgba(255,100,100,0.7)",
        colorBg: "rgba(255,100,100,0.06)",
    },
    {
        name: "Synth",
        icon: Icons.synth,
        color: "rgba(200,100,255,0.7)",
        colorBg: "rgba(200,100,255,0.06)",
    },
    {
        name: "Guitar",
        icon: Icons.guitar,
        color: "rgba(255,200,50,0.7)",
        colorBg: "rgba(255,200,50,0.06)",
    },
]

// ── Genres ──
export const genres = [
    {
        name: "Normal",
        bg: "linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.15))",
        glow: "100,160,255",
    },
    {
        name: "Jazz",
        bg: "linear-gradient(135deg, rgba(140,80,220,0.2), rgba(100,50,180,0.15))",
        glow: "120,60,200",
    },
    {
        name: "Reggae",
        bg: "linear-gradient(135deg, rgba(76,175,80,0.2), rgba(255,235,59,0.15))",
        glow: "50,200,80",
    },
    {
        name: "Electronic",
        bg: "linear-gradient(135deg, rgba(0,188,212,0.2), rgba(156,39,176,0.15))",
        glow: "30,60,180",
    },
]

// ── Track interface ──
export interface TrackItem {
    id: string
    name: string
    icon: React.ReactNode
    color: string
    colorBg: string
    seed: number
    startFrac: number
    widthFrac: number
    isPlaceholder: boolean
    row: number
}

// ── Deterministic waveform ──
function seededRandom(seed: number) {
    let s = seed
    return () => {
        s = (s * 16807 + 0) % 2147483647
        return (s - 1) / 2147483646
    }
}

export function WaveformSVG({
    color,
    seed,
    width,
    height,
}: {
    color: string
    seed: number
    width: number
    height: number
}) {
    const rng = seededRandom(seed)
    const step = 3
    const mid = height / 2
    let d = ""
    for (let x = 0; x < width; x += step) {
        const env = 0.4 + 0.6 * Math.sin((x / width) * Math.PI)
        const burst = rng() > 0.87 ? 1.3 : 1
        const amp = rng() * env * (height * 0.42) * burst
        d += `M${x},${mid - amp}V${mid + amp}`
    }
    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
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

// ── Animated noise waveform ──
export function NoiseWaveform({
    width,
    height,
    color = "rgba(139,92,246,0.6)",
    settling = false,
}: {
    width: number
    height: number
    color?: string
    settling?: boolean
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const animRef = useRef<number>(0)
    const prevAmps = useRef<number[] | null>(null)
    const startRef = useRef(Date.now())

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
        const lineCount = Math.floor(width / 4)
        const centerY = height / 2
        if (!prevAmps.current) prevAmps.current = new Array(lineCount).fill(0)

        const animate = () => {
            const elapsed = (Date.now() - startRef.current) / 1000
            ctx.clearRect(0, 0, width, height)
            ctx.strokeStyle = color
            ctx.lineWidth = 2
            for (let i = 0; i < lineCount; i++) {
                const x = (i / lineCount) * width
                const prev = prevAmps.current![i] || 0
                let amp: number
                if (settling) {
                    const p = Math.min(elapsed / 0.5, 1)
                    const ease = 1 - Math.pow(1 - p, 3)
                    amp = prev * (1 - ease)
                } else {
                    const target = (Math.random() - 0.5) * height * 0.6
                    amp = prev + (target - prev) * 0.08
                }
                prevAmps.current![i] = amp
                ctx.beginPath()
                ctx.moveTo(x, centerY - amp)
                ctx.lineTo(x, centerY + amp)
                ctx.stroke()
            }
            if (!settling || elapsed < 0.5)
                animRef.current = requestAnimationFrame(animate)
        }
        animate()
        return () => cancelAnimationFrame(animRef.current)
    }, [width, height, color, settling])

    useEffect(() => {
        if (settling) startRef.current = Date.now()
    }, [settling])

    return (
        <canvas ref={canvasRef} style={{ width, height, display: "block" }} />
    )
}

// ── Glass styling helper ──
export const glassPanel = (
    glowRGB: string,
    opts?: { blur?: number; bg?: string; radius?: number }
) => ({
    background: opts?.bg || "rgba(10,10,16,0.5)",
    backdropFilter: `blur(${opts?.blur || 32}px) saturate(170%)`,
    WebkitBackdropFilter: `blur(${opts?.blur || 32}px) saturate(170%)`,
    borderRadius: opts?.radius || 12,
    border: "1px solid rgba(255,255,255,0.1)",
    boxShadow: `0 6px 24px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 40px rgba(${glowRGB},0.06)`,
})

// ── SVG Filters ──
export function GlassFilters() {
    return (
        <svg width="0" height="0" style={{ position: "absolute" }}>
            <defs>
                <filter
                    id="liquid-edge"
                    x="-5%"
                    y="-5%"
                    width="110%"
                    height="110%"
                >
                    <feTurbulence
                        type="fractalNoise"
                        baseFrequency="0.015"
                        numOctaves="3"
                        seed="2"
                        result="noise"
                    />
                    <feDisplacementMap
                        in="SourceGraphic"
                        in2="noise"
                        scale="3"
                        xChannelSelector="R"
                        yChannelSelector="G"
                    />
                </filter>
                <filter id="glass-noise">
                    <feTurbulence
                        type="fractalNoise"
                        baseFrequency="0.8"
                        numOctaves="4"
                        stitchTiles="stitch"
                        result="noise"
                    />
                    <feColorMatrix
                        type="saturate"
                        values="0"
                        in="noise"
                        result="mono"
                    />
                    <feBlend in="SourceGraphic" in2="mono" mode="overlay" />
                </filter>
            </defs>
        </svg>
    )
}

// ── Slide wrapper — consistent frame for all slides ──
/**
 * SlideFrame — visual chrome (glow + animation area) for an individual slide.
 *
 * The `headline` and `copy` props are still accepted for backwards compat
 * with the existing slide1/3/4/5 call sites, but the bottom text overlay
 * has been removed: the headline + copy are now rendered ONCE in Home.js
 * outside the slideshow library so they don't slide along with the canvas.
 * See Home.js SLIDE_LABELS for the (single) source of those strings now.
 */
export function SlideFrame({
    children,
    glowRGB,
    headline: _headline,
    copy: _copy,
    width,
    height,
    style,
}: {
    children: React.ReactNode
    glowRGB: string
    headline?: string
    copy?: string
    width: number
    height: number
    style?: React.CSSProperties
}) {
    return (
        <div
            style={{
                width,
                height,
                position: "relative",
                overflow: "hidden",
                // Cream workbench canvas (matches --wb-bg in theme-workbench.css).
                // The inner animation elements keep their own cool-toned
                // backgrounds; only the outer frame flips to cream.
                background: "#e8e6e1",
                fontFamily:
                    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                userSelect: "none",
                ...style,
            }}
        >
            <GlassFilters />

            {/* Ambient glow */}
            <div
                style={{
                    position: "absolute",
                    inset: -40,
                    background: `radial-gradient(ellipse at 50% 40%, rgba(${glowRGB},0.2) 0%, rgba(${glowRGB},0.08) 30%, rgba(${glowRGB},0.03) 55%, transparent 80%)`,
                    pointerEvents: "none",
                    transition: "background 1.5s ease",
                }}
            />

            {/* Animation area — keeps the same top:40 / bottom:120 reserve as
                when the text overlay lived inside the frame, so individual
                slide layouts don't need to re-tune themselves. The Home-level
                overlay sits over that bottom 120px region. */}
            <div
                style={{
                    position: "absolute",
                    top: 40,
                    left: 40,
                    right: 40,
                    bottom: 120,
                }}
            >
                {children}
            </div>
        </div>
    )
}

// ── Transport bar (reusable mini version) ──
export function TransportBar({
    glowRGB,
    isPlaying = true,
    time = "0:12",
}: {
    glowRGB: string
    isPlaying?: boolean
    time?: string
}) {
    return (
        <div
            style={{
                height: 32,
                display: "flex",
                alignItems: "center",
                padding: "0 10px",
                background: "rgba(8,8,14,0.3)",
                backdropFilter: "blur(24px) saturate(170%)",
                WebkitBackdropFilter: "blur(24px) saturate(170%)",
                borderBottom: "1px solid rgba(255,255,255,0.08)",
                borderRadius: "12px 12px 0 0",
                gap: 6,
            }}
        >
            <div
                style={{
                    width: 24,
                    height: 24,
                    borderRadius: 6,
                    border: `1px solid ${isPlaying ? `rgba(${glowRGB},0.5)` : "rgba(255,255,255,0.1)"}`,
                    background: isPlaying
                        ? `rgba(${glowRGB},0.2)`
                        : "rgba(255,255,255,0.06)",
                    color: isPlaying ? "#fff" : "#ccc",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                }}
            >
                {isPlaying ? Icons.pause : Icons.play}
            </div>
            <div
                style={{
                    width: 24,
                    height: 24,
                    borderRadius: 6,
                    border: "1px solid rgba(255,255,255,0.1)",
                    background: "rgba(255,255,255,0.06)",
                    color: "#ccc",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                }}
            >
                {Icons.stop}
            </div>
            <div
                style={{
                    color: "#fff",
                    fontSize: 11,
                    fontWeight: 600,
                    minWidth: 40,
                    padding: "0 6px",
                    background: "rgba(6,6,12,0.2)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 6,
                    height: 24,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontVariantNumeric: "tabular-nums",
                }}
            >
                {time}
            </div>
            <div style={{ flex: 1 }} />
            <div
                style={{
                    padding: "3px 8px",
                    borderRadius: 6,
                    fontSize: 10,
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    color: C.textMuted,
                    fontVariantNumeric: "tabular-nums",
                }}
            >
                120 BPM
            </div>
        </div>
    )
}

// ── Timeline ticks ──
export function Timeline({
    duration = 30,
    labelW = 100,
}: {
    duration?: number
    labelW?: number
}) {
    const ticks = []
    const interval = duration <= 15 ? 2 : duration <= 60 ? 5 : 10
    for (let t = 0; t <= duration; t += interval) {
        ticks.push({ time: t, pct: `${(t / duration) * 100}%` })
    }
    return (
        <div
            style={{
                height: 24,
                display: "flex",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                background: "rgba(6,6,12,0.15)",
                backdropFilter: "blur(20px) saturate(150%)",
            }}
        >
            <div
                style={{
                    width: labelW,
                    minWidth: labelW,
                    background: "rgba(6,6,12,0.15)",
                    borderRight: "1px solid rgba(255,255,255,0.06)",
                    display: "flex",
                    alignItems: "center",
                    paddingLeft: 10,
                }}
            >
                <span style={{ fontSize: 9, color: C.textDim }}>
                    + Add Track
                </span>
            </div>
            <div
                style={{
                    flex: 1,
                    position: "relative",
                    background: "rgba(6,6,10,0.1)",
                }}
            >
                {ticks.map((tick) => (
                    <div
                        key={tick.time}
                        style={{
                            position: "absolute",
                            left: tick.pct,
                            top: 0,
                            height: "100%",
                        }}
                    >
                        <span
                            style={{
                                position: "absolute",
                                top: 4,
                                left: 3,
                                color: C.textDim,
                                fontSize: 8,
                                whiteSpace: "nowrap",
                            }}
                        >
                            {tick.time}s
                        </span>
                        <div
                            style={{
                                position: "absolute",
                                bottom: 0,
                                left: 0,
                                width: 1,
                                height: 8,
                                background: "#333",
                            }}
                        />
                    </div>
                ))}
            </div>
        </div>
    )
}

// ── Single track row (rendered state) ──
export function TrackRow({
    track,
    labelW = 100,
    trackHeight = 56,
    waveW = 300,
    glowRGB,
}: {
    track: TrackItem
    labelW?: number
    trackHeight?: number
    waveW?: number
    glowRGB?: string
}) {
    return (
        <div
            style={{
                position: "relative",
                height: trackHeight,
                borderBottom: "1px solid rgba(255,255,255,0.04)",
            }}
        >
            {/* Label */}
            <div
                style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: labelW,
                    display: "flex",
                    alignItems: "center",
                    padding: "0 6px",
                    gap: 6,
                    background: "rgba(8,8,14,0.2)",
                    backdropFilter: "blur(20px) saturate(150%)",
                    borderRight: "1px solid rgba(255,255,255,0.06)",
                }}
            >
                <div
                    style={{ color: track.color, opacity: 0.7, flexShrink: 0 }}
                >
                    {track.icon}
                </div>
                <span
                    style={{ fontSize: 10, color: C.textSec, fontWeight: 500 }}
                >
                    {track.name}
                </span>
            </div>
            {/* Clip */}
            <div
                style={{
                    position: "absolute",
                    top: 3,
                    left: `calc(${labelW}px + ${track.startFrac * 100}% * (1 - ${labelW}/${waveW + labelW}) + 3px)`,
                    width: `calc(${track.widthFrac * 100}% * (1 - ${labelW}/${waveW + labelW}) - 6px)`,
                    height: trackHeight - 6,
                    overflow: "hidden",
                    borderRadius: 8,
                }}
            >
                {track.isPlaceholder ? (
                    <div
                        style={{
                            position: "absolute",
                            inset: 0,
                            borderRadius: 8,
                            background: `rgba(${glowRGB || "139,92,246"},0.04)`,
                            border: `1px solid rgba(${glowRGB || "139,92,246"},0.12)`,
                        }}
                    >
                        <NoiseWaveform
                            width={Math.max(50, track.widthFrac * waveW - 12)}
                            height={trackHeight - 10}
                            color={track.color}
                        />
                    </div>
                ) : (
                    <div
                        style={{
                            position: "absolute",
                            inset: 0,
                            background: `linear-gradient(180deg, ${track.colorBg}, ${track.colorBg.replace("0.06", "0.02")})`,
                            backdropFilter: "blur(14px) saturate(150%)",
                            borderLeft: `2px solid ${track.color}`,
                            borderRight: `1px solid ${track.color.replace("0.7", "0.12")}`,
                            borderTop: `1px solid ${track.color.replace("0.7", "0.15")}`,
                            borderBottom: `1px solid ${track.color.replace("0.7", "0.05")}`,
                            borderRadius: 8,
                            boxShadow: `0 4px 16px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.08)`,
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
                            <WaveformSVG
                                color={track.color}
                                seed={track.seed}
                                width={Math.max(
                                    50,
                                    track.widthFrac * waveW - 18
                                )}
                                height={trackHeight - 18}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
