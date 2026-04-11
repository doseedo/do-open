/**
 * Slide 4 — "Score a video in minutes."
 * Animation: Video area loads, scene markers appear on timeline with thumbnails + scene numbers,
 * then tracks appear and markers become simple arrows.
 * Loops every 9 seconds.
 */
import React, { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { addPropertyControls, ControlType } from "./framer-stub"
import {
    C,
    Icons,
    instrumentPool,
    WaveformSVG,
    NoiseWaveform,
    glassPanel,
    SlideFrame,
    TransportBar,
    Timeline,
} from "./shared"

const GLOW = "30,60,180"
const CYCLE = 9000
const VIDEO_URL =
    "https://storage.googleapis.com/audiocraft-411005.appspot.com/assets/sitedemo.mp4"

const scoreTracks = [
    { ...instrumentPool[1], name: "Strings", seed: 510, row: 0 },
    { ...instrumentPool[2], name: "Piano", seed: 620, row: 1 },
    { ...instrumentPool[4], name: "Drums", seed: 730, row: 2 },
    { ...instrumentPool[5], name: "Bass", seed: 840, row: 3 },
]

// Scene markers positioned along the timeline (as fraction 0–1)
const sceneMarkers = [
    { id: 1, pos: 0.05, color: "rgba(100,180,255,0.8)", label: "Sc 1" },
    { id: 2, pos: 0.28, color: "rgba(255,160,60,0.8)", label: "Sc 2" },
    { id: 3, pos: 0.52, color: "rgba(120,220,120,0.8)", label: "Sc 3" },
    { id: 4, pos: 0.76, color: "rgba(255,100,160,0.8)", label: "Sc 4" },
]

function Slide4_ScoreVideo(props: {
    width?: number
    height?: number
    style?: React.CSSProperties
}) {
    const { width = 800, height = 500, style } = props
    const [phase, setPhase] = useState<
        "video" | "analyzing" | "generating" | "scoring" | "done"
    >("video")
    const [revealedTracks, setRevealedTracks] = useState<number[]>([])
    const videoRef = useRef<HTMLVideoElement>(null)
    const cycleRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const labelW = 90
    const trackH = 44

    useEffect(() => {
        const run = () => {
            setPhase("video")
            setRevealedTracks([])
            if (videoRef.current) {
                videoRef.current.currentTime = 0
                videoRef.current.play().catch(() => {})
            }

            // Show video for 1.5s
            setTimeout(() => setPhase("analyzing"), 1500)
            // Scene markers visible during analyzing, then start generating
            setTimeout(() => setPhase("generating"), 3200)
            setTimeout(() => setPhase("scoring"), 3800)

            // Stagger reveal tracks
            scoreTracks.forEach((_, i) => {
                setTimeout(
                    () => {
                        setRevealedTracks((prev) => [...prev, i])
                    },
                    4200 + i * 400
                )
            })

            setTimeout(
                () => setPhase("done"),
                4200 + scoreTracks.length * 400 + 400
            )
            cycleRef.current = setTimeout(run, CYCLE)
        }
        run()
        return () => {
            if (cycleRef.current) clearTimeout(cycleRef.current)
        }
    }, [])

    const showTracks = phase === "scoring" || phase === "done"

    return (
        <SlideFrame
            glowRGB={GLOW}
            headline="Personalized, adaptive music"
            copy="Adapt exiting music to picture, or generate new scores tailored for your visual media. With track level key framing, you can polish to perfection with ease."
            width={width}
            height={height}
            style={style}
        >
            <div
                style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    flexDirection: "column",
                    gap: 8,
                }}
            >
                {/* Video area — top half */}
                <div
                    style={{
                        flex: 1,
                        ...glassPanel(GLOW, { radius: 14 }),
                        overflow: "hidden",
                        position: "relative",
                    }}
                >
                    {/* Tab bar */}
                    <div
                        style={{
                            display: "flex",
                            gap: 0,
                            borderBottom: "1px solid rgba(255,255,255,0.06)",
                            background: "rgba(6,6,12,0.25)",
                            position: "relative",
                            zIndex: 2,
                        }}
                    >
                        {["Image", "FX", "Video", "MIDI", "Audio"].map(
                            (tab, i) => (
                                <div
                                    key={tab}
                                    style={{
                                        padding: "6px 12px",
                                        fontSize: 9,
                                        fontWeight: 500,
                                        color:
                                            i === 2
                                                ? `rgba(${GLOW},1)`
                                                : C.textMuted,
                                        borderBottom:
                                            i === 2
                                                ? `2px solid rgba(${GLOW},1)`
                                                : "2px solid transparent",
                                        background:
                                            i === 2
                                                ? `rgba(${GLOW},0.06)`
                                                : "transparent",
                                    }}
                                >
                                    {tab}
                                </div>
                            )
                        )}
                    </div>

                    {/* Video */}
                    <video
                        ref={videoRef}
                        src={VIDEO_URL}
                        playsInline
                        muted
                        loop
                        preload="auto"
                        style={{
                            position: "absolute",
                            top: 28,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            width: "100%",
                            height: "calc(100% - 28px)",
                            objectFit: "cover",
                            opacity: 0.7,
                        }}
                    />

                    {/* Video overlay gradient */}
                    <div
                        style={{
                            position: "absolute",
                            bottom: 0,
                            left: 0,
                            right: 0,
                            height: 60,
                            background:
                                "linear-gradient(to top, rgba(5,5,8,0.9), transparent)",
                            zIndex: 1,
                        }}
                    />

                    {/* Scene markers over video — bottom of video area */}
                    <AnimatePresence>
                        {(phase === "analyzing" || phase === "generating") && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.4 }}
                                style={{
                                    position: "absolute",
                                    bottom: 8,
                                    left: 12,
                                    right: 12,
                                    display: "flex",
                                    justifyContent: "space-around",
                                    zIndex: 4,
                                }}
                            >
                                {sceneMarkers.map((marker, mi) => (
                                    <motion.div
                                        key={marker.id}
                                        initial={{
                                            opacity: 0,
                                            y: 16,
                                            scale: 0.85,
                                        }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        transition={{
                                            duration: 0.35,
                                            delay: mi * 0.12,
                                            ease: [0.2, 0.8, 0.2, 1],
                                        }}
                                        style={{
                                            display: "flex",
                                            flexDirection: "column",
                                            alignItems: "center",
                                            gap: 3,
                                        }}
                                    >
                                        {/* Video clip thumbnail */}
                                        <div
                                            style={{
                                                width: 48,
                                                height: 32,
                                                borderRadius: 5,
                                                background: `linear-gradient(135deg, ${marker.color.replace("0.8", "0.25")}, ${marker.color.replace("0.8", "0.08")})`,
                                                border: `1px solid ${marker.color.replace("0.8", "0.5")}`,
                                                boxShadow: `0 2px 10px rgba(0,0,0,0.5), 0 0 8px ${marker.color.replace("0.8", "0.15")}`,
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                overflow: "hidden",
                                                position: "relative",
                                            }}
                                        >
                                            {/* Film frame lines */}
                                            <div
                                                style={{
                                                    position: "absolute",
                                                    left: 0,
                                                    top: 0,
                                                    bottom: 0,
                                                    width: 4,
                                                    background: `repeating-linear-gradient(180deg, ${marker.color.replace("0.8", "0.25")} 0px, ${marker.color.replace("0.8", "0.25")} 2px, transparent 2px, transparent 4px)`,
                                                }}
                                            />
                                            <div
                                                style={{
                                                    position: "absolute",
                                                    right: 0,
                                                    top: 0,
                                                    bottom: 0,
                                                    width: 4,
                                                    background: `repeating-linear-gradient(180deg, ${marker.color.replace("0.8", "0.25")} 0px, ${marker.color.replace("0.8", "0.25")} 2px, transparent 2px, transparent 4px)`,
                                                }}
                                            />
                                            <svg
                                                width={14}
                                                height={14}
                                                viewBox="0 0 24 24"
                                                fill="none"
                                            >
                                                <polygon
                                                    points="8 5 19 12 8 19"
                                                    fill={marker.color}
                                                    stroke="none"
                                                />
                                            </svg>
                                        </div>
                                        {/* Scene label */}
                                        <span
                                            style={{
                                                fontSize: 8,
                                                fontWeight: 700,
                                                color: marker.color,
                                                textTransform: "uppercase",
                                                letterSpacing: 0.5,
                                            }}
                                        >
                                            {marker.label}
                                        </span>
                                    </motion.div>
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Import indicator */}
                    <AnimatePresence>
                        {phase === "video" && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    position: "absolute",
                                    bottom: 12,
                                    left: "50%",
                                    transform: "translateX(-50%)",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                    padding: "6px 14px",
                                    borderRadius: 16,
                                    background: `rgba(${GLOW},0.2)`,
                                    backdropFilter: "blur(16px)",
                                    border: `1px solid rgba(${GLOW},0.3)`,
                                    zIndex: 2,
                                }}
                            >
                                <div style={{ color: `rgba(${GLOW},0.9)` }}>
                                    {Icons.film}
                                </div>
                                <span
                                    style={{
                                        fontSize: 10,
                                        color: "#fff",
                                        fontWeight: 600,
                                    }}
                                >
                                    Video imported
                                </span>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Analyzing overlay */}
                    <AnimatePresence>
                        {phase === "analyzing" && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    position: "absolute",
                                    bottom: 12,
                                    left: "50%",
                                    transform: "translateX(-50%)",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 8,
                                    padding: "6px 14px",
                                    borderRadius: 16,
                                    background: `rgba(${GLOW},0.2)`,
                                    backdropFilter: "blur(16px)",
                                    border: `1px solid rgba(${GLOW},0.3)`,
                                    zIndex: 2,
                                }}
                            >
                                <div style={{ color: `rgba(${GLOW},0.9)` }}>
                                    {Icons.film}
                                </div>
                                <span
                                    style={{
                                        fontSize: 10,
                                        color: "#fff",
                                        fontWeight: 600,
                                    }}
                                >
                                    Analyzing scenes...
                                </span>
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{
                                        duration: 1,
                                        repeat: Infinity,
                                        ease: "linear",
                                    }}
                                    style={{
                                        width: 12,
                                        height: 12,
                                        borderRadius: "50%",
                                        border: `2px solid rgba(${GLOW},0.2)`,
                                        borderTopColor: `rgba(${GLOW},0.8)`,
                                    }}
                                />
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Generating overlay */}
                    <AnimatePresence>
                        {phase === "generating" && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    background: "rgba(0,0,0,0.4)",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    zIndex: 3,
                                }}
                            >
                                <div
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                        padding: "10px 20px",
                                        borderRadius: 20,
                                        background: `rgba(${GLOW},0.15)`,
                                        backdropFilter: "blur(20px)",
                                        border: `1px solid rgba(${GLOW},0.3)`,
                                    }}
                                >
                                    <div style={{ color: `rgba(${GLOW},0.9)` }}>
                                        {Icons.wand}
                                    </div>
                                    <span
                                        style={{
                                            fontSize: 11,
                                            color: "#fff",
                                            fontWeight: 600,
                                        }}
                                    >
                                        Scoring video...
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
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* DAW area — bottom half */}
                <div
                    style={{
                        height: scoreTracks.length * trackH + 56,
                        flexShrink: 0,
                        ...glassPanel(GLOW, { radius: 14 }),
                        overflow: "hidden",
                        display: "flex",
                        flexDirection: "column",
                    }}
                >
                    <TransportBar
                        glowRGB={GLOW}
                        isPlaying={phase === "done"}
                        time={phase === "done" ? "0:12" : "0:00"}
                    />

                    <Timeline duration={30} labelW={labelW} />

                    <div
                        style={{
                            flex: 1,
                            position: "relative",
                            overflow: "hidden",
                        }}
                    >
                        {/* Row backgrounds */}
                        {scoreTracks.map((_, i) => (
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

                        {/* Stem tracks — no slide-in animation */}
                        {showTracks &&
                            scoreTracks.map((track, i) => {
                                const isRevealed = revealedTracks.includes(i)
                                return (
                                    <div key={track.name}>
                                        {/* Label — appears instantly, no x slide */}
                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{
                                                duration: 0.3,
                                                delay: i * 0.06,
                                            }}
                                            style={{
                                                position: "absolute",
                                                top: i * trackH,
                                                left: 0,
                                                width: labelW,
                                                height: trackH,
                                                display: "flex",
                                                alignItems: "center",
                                                padding: "0 6px",
                                                gap: 5,
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
                                                    color: track.color,
                                                    opacity: 0.7,
                                                }}
                                            >
                                                {track.icon}
                                            </div>
                                            <span
                                                style={{
                                                    fontSize: 9,
                                                    color: C.textSec,
                                                    fontWeight: 500,
                                                }}
                                            >
                                                {track.name}
                                            </span>
                                        </motion.div>

                                        {/* Clip */}
                                        <div
                                            style={{
                                                position: "absolute",
                                                top: i * trackH + 3,
                                                left: labelW + 6,
                                                right: 6,
                                                height: trackH - 6,
                                                overflow: "hidden",
                                                borderRadius: 7,
                                            }}
                                        >
                                            {isRevealed ? (
                                                <motion.div
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                    transition={{
                                                        duration: 0.6,
                                                        ease: "easeOut",
                                                    }}
                                                    style={{
                                                        position: "absolute",
                                                        inset: 0,
                                                        background: `linear-gradient(180deg, ${track.colorBg}, ${track.colorBg.replace("0.06", "0.02")})`,
                                                        backdropFilter:
                                                            "blur(14px) saturate(150%)",
                                                        borderLeft: `2px solid ${track.color}`,
                                                        borderRadius: 7,
                                                        boxShadow: `0 3px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08)`,
                                                    }}
                                                >
                                                    <div
                                                        style={{
                                                            position:
                                                                "absolute",
                                                            left: 5,
                                                            top: "50%",
                                                            transform:
                                                                "translateY(-50%)",
                                                        }}
                                                    >
                                                        <WaveformSVG
                                                            color={track.color}
                                                            seed={track.seed}
                                                            width={450}
                                                            height={trackH - 18}
                                                        />
                                                    </div>
                                                </motion.div>
                                            ) : (
                                                <div
                                                    style={{
                                                        position: "absolute",
                                                        inset: 0,
                                                        borderRadius: 7,
                                                        background: `rgba(${GLOW},0.04)`,
                                                        border: `1px solid rgba(${GLOW},0.12)`,
                                                    }}
                                                >
                                                    <NoiseWaveform
                                                        width={450}
                                                        height={trackH - 10}
                                                        color={track.color}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )
                            })}

                        {/* Playhead */}
                        {phase === "done" && (
                            <motion.div
                                initial={{ left: "0%" }}
                                animate={{ left: "40%" }}
                                transition={{ duration: 4, ease: "linear" }}
                                style={{
                                    position: "absolute",
                                    top: 0,
                                    width: 1.5,
                                    height: "100%",
                                    background: "#fff",
                                    boxShadow: "0 0 8px rgba(255,255,255,0.25)",
                                    zIndex: 50,
                                    marginLeft: labelW,
                                }}
                            />
                        )}
                    </div>
                </div>
            </div>
        </SlideFrame>
    )
}

addPropertyControls(Slide4_ScoreVideo, {
    width: { type: ControlType.Number, defaultValue: 800, min: 400, max: 1400 },
    height: { type: ControlType.Number, defaultValue: 500, min: 300, max: 800 },
})

export default Slide4_ScoreVideo
