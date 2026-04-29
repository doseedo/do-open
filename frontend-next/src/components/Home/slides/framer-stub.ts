/**
 * Stub for the `framer` runtime package (https://www.framer.com/docs/).
 * The slides in this directory were authored inside the Framer IDE and use
 * `addPropertyControls` / `ControlType` to expose design-time tweaks in the
 * Framer canvas. Those APIs are a no-op inside the SPA; this file lets the
 * slides import from `"./framer-stub"` with zero behavior change.
 */

export const addPropertyControls = (..._args: unknown[]): void => {
  /* no-op outside Framer */
};

export const ControlType = {
  Number: "number",
  Boolean: "boolean",
  String: "string",
  Enum: "enum",
  Color: "color",
  Image: "image",
  File: "file",
  Array: "array",
  Object: "object",
  ComponentInstance: "componentInstance",
  EventHandler: "eventHandler",
  FusedNumber: "fusedNumber",
  SegmentedEnum: "segmentedEnum",
  Transition: "transition",
  Link: "link",
  PageScope: "pageScope",
  Border: "border",
  BoxShadow: "boxShadow",
  Padding: "padding",
  Margin: "margin",
  Color2: "color",
} as const;
