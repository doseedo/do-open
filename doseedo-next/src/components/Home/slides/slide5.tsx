/**
 * Slide 5 — "Everything stays editable."
 * Animation: Tracks play back, then a track gets modified mid-playback,
 * waveform reshapes, demonstrating non-destructive editing.
 * Loops every 9 seconds.
 */
import React, { useState, useEffect, useRef, useCallback } from "react"
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

const GLOW = "100,160,255"
const CYCLE = 9000

const editTracks = [
    { ...instrumentPool[0], name: "Vocals", seed: 111, row: 0 },
    { ...instrumentPool[4], name: "Drums", seed: 222, row: 1 },
    { ...instrumentPool[5], name: "Bass", seed: 333, row: 2 },
    { ...instrumentPool[6], name: "Synth", seed: 444, row: 3 },
]

function Slide5_Editable(props: {
    width?: number
    height?: number
    style?: React.CSSProperties
}) {
    const { width = 800, height = 500, style } = props
    const [phase, setPhase] = useState<
        "playing" | "selecting" | "editing" | "reshaped" | "exporting"
    >("playing")
    const [playhead, setPlayhead] = useState(0)
    const [editTarget, setEditTarget] = useState(2) // Bass track gets edited
    const [editSeed, setEditSeed] = useState(333)
    const playRef = useRef<number>(0)
    const startRef = useRef(Date.now())
    const cycleRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const labelW = 100
    const trackH = 52

    // Playhead animation
    const animatePlayhead = useCallback(() => {
        const elapsed = (Date.now() - startRef.current) / 1000
        setPlayhead((elapsed % 30) / 30)
        playRef.current = requestAnimationFrame(animatePlayhead)
    }, [])

    useEffect(() => {
        startRef.current = Date.now()
        playRef.current = requestAnimationFrame(animatePlayhead)
        return () => cancelAnimationFrame(playRef.current)
    }, [animatePlayhead])

    // Phase cycle
    useEffect(() => {
        const run = () => {
            setPhase("playing")
            startRef.current = Date.now()
            setEditSeed(333)

            // Select a track after 2s
            setTimeout(() => setPhase("selecting"), 2000)

            // Start editing at 3.5s
            setTimeout(() => {
                setPhase("editing")
                setEditSeed(888) // New waveform seed
            }, 3500)

            // Show reshaped at 5s
            setTimeout(() => setPhase("reshaped"), 5000)

            // Show export button at 6.5s
            setTimeout(() => setPhase("exporting"), 6500)

            cycleRef.current = setTimeout(run, CYCLE)
        }
        run()
        return () => {
            if (cycleRef.current) clearTimeout(cycleRef.current)
        }
    }, [])

    const isSelected =
        phase === "selecting" ||
        phase === "editing" ||
        phase === "reshaped" ||
        phase === "exporting"
    const isEditing = phase === "editing"

    return (
        <SlideFrame
            glowRGB={GLOW}
            headline="Everything stays editable."
            copy="Nothing is ever frozen. Come back and reshape it anytime."
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
                <TransportBar
                    glowRGB={GLOW}
                    isPlaying={true}
                    time={`0:${String(Math.floor(playhead * 30)).padStart(2, "0")}`}
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
                    {editTracks.map((_, i) => (
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

                    {/* Tracks */}
                    {editTracks.map((track, i) => {
                        const isTarget = i === editTarget
                        const currentSeed = isTarget ? editSeed : track.seed
                        const showNoise = isTarget && isEditing

                        return (
                            <div key={track.name}>
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
                                        background:
                                            isTarget && isSelected
                                                ? `rgba(${GLOW},0.06)`
                                                : "rgba(8,8,14,0.4)",
                                        backdropFilter: "blur(20px)",
                                        borderRight:
                                            "1px solid rgba(255,255,255,0.06)",
                                        borderBottom:
                                            "1px solid rgba(255,255,255,0.04)",
                                        zIndex: 2,
                                        transition: "background 0.3s ease",
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
                                            fontSize: 10,
                                            color: C.textSec,
                                            fontWeight: 500,
                                        }}
                                    >
                                        {track.name}
                                    </span>
                                    {isTarget && isSelected && (
                                        <motion.div
                                            initial={{ opacity: 0, scale: 0.5 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            style={{
                                                marginLeft: "auto",
                                                width: 6,
                                                height: 6,
                                                borderRadius: "50%",
                                                background: `rgba(${GLOW},0.8)`,
                                                boxShadow: `0 0 6px rgba(${GLOW},0.5)`,
                                            }}
                                        />
                                    )}
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
                                    {/* Selection highlight */}
                                    {isTarget && isSelected && (
                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            style={{
                                                position: "absolute",
                                                inset: -2,
                                                borderRadius: 10,
                                                border: `1.5px solid rgba(${GLOW},0.5)`,
                                                boxShadow: `0 0 12px rgba(${GLOW},0.15), inset 0 0 12px rgba(${GLOW},0.05)`,
                                                pointerEvents: "none",
                                                zIndex: 5,
                                            }}
                                        />
                                    )}

                                    {showNoise ? (
                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            style={{
                                                position: "absolute",
                                                inset: 0,
                                                borderRadius: 8,
                                                background: `rgba(${GLOW},0.04)`,
                                                border: `1px solid rgba(${GLOW},0.15)`,
                                            }}
                                        >
                                            <NoiseWaveform
                                                width={500}
                                                height={trackH - 10}
                                                color={track.color}
                                            />
                                        </motion.div>
                                    ) : (
                                        <motion.div
                                            key={currentSeed}
                                            initial={
                                                isTarget && phase === "reshaped"
                                                    ? {
                                                          opacity: 0,
                                                          scaleX: 0.5,
                                                      }
                                                    : false
                                            }
                                            animate={{ opacity: 1, scaleX: 1 }}
                                            transition={{
                                                duration: 0.5,
                                                ease: [0.2, 0.8, 0.2, 1],
                                            }}
                                            style={{
                                                position: "absolute",
                                                inset: 0,
                                                transformOrigin: "left center",
                                                background: `linear-gradient(180deg, ${track.colorBg}, ${track.colorBg.replace("0.06", "0.02")})`,
                                                backdropFilter:
                                                    "blur(14px) saturate(150%)",
                                                borderLeft: `2px solid ${track.color}`,
                                                borderRight: `1px solid ${track.color.replace("0.7", "0.12")}`,
                                                borderTop: `1px solid ${track.color.replace("0.7", "0.15")}`,
                                                borderBottom: `1px solid ${track.color.replace("0.7", "0.05")}`,
                                                borderRadius: 8,
                                                boxShadow: `0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08)`,
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
                                                    color={track.color}
                                                    seed={currentSeed}
                                                    width={500}
                                                    height={trackH - 20}
                                                />
                                            </div>
                                        </motion.div>
                                    )}
                                </div>
                            </div>
                        )
                    })}

                    {/* Playhead */}
                    <div
                        style={{
                            position: "absolute",
                            top: 0,
                            left: `calc(${labelW}px + ${playhead * 100}% * (1 - ${labelW / 800}))`,
                            width: 1.5,
                            height: "100%",
                            background: "#fff",
                            boxShadow: "0 0 8px rgba(255,255,255,0.25)",
                            zIndex: 50,
                            pointerEvents: "none",
                        }}
                    />
                    <div
                        style={{
                            position: "absolute",
                            top: -1,
                            left: `calc(${labelW}px + ${playhead * 100}% * (1 - ${labelW / 800}) - 5px)`,
                            width: 0,
                            height: 0,
                            borderLeft: "5px solid transparent",
                            borderRight: "5px solid transparent",
                            borderTop: "8px solid #fff",
                            zIndex: 51,
                            pointerEvents: "none",
                        }}
                    />

                    {/* Edit action badge */}
                    <AnimatePresence>
                        {isEditing && (
                            <motion.div
                                initial={{ opacity: 0, y: 10, scale: 0.9 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                style={{
                                    position: "absolute",
                                    top: editTarget * trackH + trackH / 2 - 14,
                                    right: 20,
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                    padding: "5px 12px",
                                    borderRadius: 14,
                                    background: `rgba(${GLOW},0.2)`,
                                    backdropFilter: "blur(16px)",
                                    border: `1px solid rgba(${GLOW},0.35)`,
                                    zIndex: 20,
                                }}
                            >
                                <div style={{ color: `rgba(${GLOW},0.9)` }}>
                                    {Icons.wand}
                                </div>
                                <span
                                    style={{
                                        fontSize: 10,
                                        color: "#fff",
                                        fontWeight: 600,
                                    }}
                                >
                                    Reshaping...
                                </span>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Export button */}
                    <AnimatePresence>
                        {phase === "exporting" && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    position: "absolute",
                                    bottom: 12,
                                    right: 12,
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                    padding: "8px 16px",
                                    borderRadius: 10,
                                    background: `linear-gradient(135deg, rgba(${GLOW},0.8), rgba(${GLOW},0.5))`,
                                    border: "1px solid rgba(255,255,255,0.15)",
                                    boxShadow: `0 4px 16px rgba(${GLOW},0.25), 0 0 20px rgba(${GLOW},0.1)`,
                                    zIndex: 20,
                                }}
                            >
                                <div style={{ color: "#fff" }}>
                                    {Icons.download}
                                </div>
                                <span
                                    style={{
                                        fontSize: 11,
                                        color: "#fff",
                                        fontWeight: 600,
                                    }}
                                >
                                    Export
                                </span>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </SlideFrame>
    )
}

addPropertyControls(Slide5_Editable, {
    width: { type: ControlType.Number, defaultValue: 800, min: 400, max: 1400 },
    height: { type: ControlType.Number, defaultValue: 500, min: 300, max: 800 },
})

export default Slide5_Editable
