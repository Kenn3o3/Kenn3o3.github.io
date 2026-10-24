// Geometry shared by the rendered fingertips and their tactile images.
// Force is relative: it controls compliant pad indentation, not calibrated N.
export const OPEN_GAP = 0.86; // distance from the centre to each sensing face
export const MAX_INDENT = 0.085;
export const PAD_SIZE = 1.1;
export const SHAPES = {
  sphere: { extent: 0.48, bottom: -0.48, description: 'a circular imprint' },
  cube: { extent: 0.42, bottom: -0.42, description: 'a square imprint' },
  cylinder: { extent: 0.46, bottom: -0.49, description: 'a vertical contact band' }
};

export function surfaceX(shape, y, z) {
  const extent = SHAPES[shape].extent;
  if (shape === 'cube') return Math.abs(y) <= extent && Math.abs(z) <= extent ? extent : null;
  if (shape === 'cylinder') {
    if (Math.abs(y) > 0.49 || Math.abs(z) > extent) return null;
    return Math.sqrt(Math.max(0, extent * extent - z * z));
  }
  const squared = extent * extent - y * y - z * z;
  return squared >= 0 ? Math.sqrt(squared) : null;
}

export function contactDepth(shape, gap, y, z) {
  const surface = surfaceX(shape, y, z);
  return surface === null ? 0 : Math.max(0, surface - gap);
}

export function gripGap(shape, force) {
  return SHAPES[shape].extent - MAX_INDENT * Math.max(0, Math.min(100, force)) / 100;
}

// Illustrative marker motion from a local indentation gradient, in image pixels.
// A finite neighbourhood represents lateral spreading of the compliant gel.
// These are displacements from the unloaded reference, not frame-to-frame flow.
export function markerDisplacement(shape, gap, y, z) {
  if (contactDepth(shape, gap, y, z) <= 0) return { x: 0, y: 0 };
  const step = 0.07;
  const load = Math.sqrt(Math.min(1, (SHAPES[shape].extent - gap) / MAX_INDENT));
  let dx = (contactDepth(shape, gap, y, z - step) - contactDepth(shape, gap, y, z + step)) * 90 * load;
  let dy = (contactDepth(shape, gap, y + step, z) - contactDepth(shape, gap, y - step, z)) * 90 * load;
  const scale = Math.min(1, 4.8 / (Math.hypot(dx, dy) || 1));
  return { x: dx * scale, y: dy * scale };
}
