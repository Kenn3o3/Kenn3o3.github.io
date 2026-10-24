import test from 'node:test';
import assert from 'node:assert/strict';
import { OPEN_GAP, MAX_INDENT, SHAPES, contactDepth, gripGap, markerDisplacement } from '../assets/js/tactile-contact.mjs';

const samples = [];
for (let y = -0.55; y <= 0.55; y += 0.01) {
  for (let z = -0.55; z <= 0.55; z += 0.01) samples.push([y, z]);
}
const area = (shape, force) => samples.filter(([y, z]) => contactDepth(shape, gripGap(shape, force), y, z) > 0).length;

for (const shape of Object.keys(SHAPES)) {
  test(`${shape}: the entire tactile image is unloaded until geometric contact`, () => {
    for (const gap of [OPEN_GAP, SHAPES[shape].extent + 0.001, SHAPES[shape].extent]) {
      assert(samples.every(([y, z]) => contactDepth(shape, gap, y, z) === 0));
    }
  });
  test(`${shape}: increasing force increases indentation without exceeding gel thickness`, () => {
    const low = contactDepth(shape, gripGap(shape, 10), 0, 0);
    const high = contactDepth(shape, gripGap(shape, 100), 0, 0);
    assert(high > low && Math.abs(high - MAX_INDENT) < 1e-10);
    assert.equal(gripGap(shape, -10), gripGap(shape, 0));
    assert.equal(gripGap(shape, 200), gripGap(shape, 100));
    assert(samples.every(([y, z]) => contactDepth(shape, gripGap(shape, 100), y, z) <= MAX_INDENT + 1e-10));
  });
}

test('a sphere has a round patch that grows with force', () => {
  const gap = gripGap('sphere', 50);
  assert(Math.abs(contactDepth('sphere', gap, 0.1, 0.2) - contactDepth('sphere', gap, 0.2, 0.1)) < 1e-10);
  assert.equal(contactDepth('sphere', gap, 0.4, 0.4), 0);
  assert(area('sphere', 80) > area('sphere', 20) * 2);
});

test('a cube retains its flat square footprint as force changes', () => {
  const gap = gripGap('cube', 50);
  assert(contactDepth('cube', gap, 0.4, 0.4) > 0);
  assert.equal(contactDepth('cube', gap, 0.43, 0), 0);
  assert.equal(contactDepth('cube', gap, 0, 0.43), 0);
  assert.equal(area('cube', 20), area('cube', 80));
});

test('a cylinder creates a vertical band, wider with more force', () => {
  const gap = gripGap('cylinder', 50);
  assert.equal(contactDepth('cylinder', gap, 0.4, 0), contactDepth('cylinder', gap, 0, 0));
  assert.equal(contactDepth('cylinder', gap, 0.5, 0), 0);
  assert.equal(contactDepth('cylinder', gap, 0, 0.4), 0);
  assert(area('cylinder', 80) > area('cylinder', 20));
});

test('marker flow is zero before contact and outside the contact patch', () => {
  for (const shape of Object.keys(SHAPES)) {
    for (const gap of [OPEN_GAP, SHAPES[shape].extent]) {
      assert(samples.every(([y, z]) => {
        const flow = markerDisplacement(shape, gap, y, z);
        return flow.x === 0 && flow.y === 0;
      }));
    }
    assert.deepEqual(markerDisplacement(shape, gripGap(shape, 100), 0.54, 0.54), { x: 0, y: 0 });
  }
});

test('marker vectors follow shape, symmetry, and applied load', () => {
  const sphereGap = gripGap('sphere', 75);
  const left = markerDisplacement('sphere', sphereGap, 0.1, -0.1);
  const right = markerDisplacement('sphere', sphereGap, 0.1, 0.1);
  assert(left.x < 0 && right.x > 0 && right.y < 0);
  assert(Math.abs(left.x + right.x) < 1e-10);
  assert.equal(left.y, right.y);
  for (const [shape, y, z] of [['sphere', 0.1, 0.1], ['cube', 0.37, 0.37], ['cylinder', 0, 0.1]]) {
    const low = markerDisplacement(shape, gripGap(shape, 25), y, z);
    const high = markerDisplacement(shape, gripGap(shape, 75), y, z);
    assert(Math.hypot(high.x, high.y) > Math.hypot(low.x, low.y), shape);
  }
  const band = markerDisplacement('cylinder', gripGap('cylinder', 75), 0, 0.1);
  assert(band.x > 0 && band.y === 0);
});
