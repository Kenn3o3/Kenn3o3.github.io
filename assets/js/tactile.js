import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { OPEN_GAP, MAX_INDENT, PAD_SIZE, SHAPES, contactDepth, gripGap, markerDisplacement } from './tactile-contact.mjs';

const study = document.querySelector('.tactile-study');
if (study) setup(study);

function setup(study) {
  const toggle = document.querySelector('.tactile-toggle');
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener('click', () => {
      const expanded = study.classList.toggle('is-expanded');
      toggle.setAttribute('aria-expanded', String(expanded));
      toggle.querySelector('span').textContent = expanded ? '−' : '+';
    });
  }
  let initialized = false;
  let setVisible = () => {};
  const observer = new IntersectionObserver(entries => {
    const visible = entries[0].isIntersecting;
    if (visible && !initialized) {
      initialized = true;
      try { setVisible = createExperiment(study); }
      catch (error) {
        study.querySelector('.grasp-fallback-note').textContent = 'The interactive view needs WebGL. Shown above: an open gripper, before contact.';
      }
    }
    setVisible(visible);
  });
  observer.observe(study);
}

function createExperiment(study) {
  const host = study.querySelector('.grasp-scene');
  const slider = study.querySelector('#grasp-force');
  const output = study.querySelector('#grasp-force-value');
  const action = study.querySelector('.grasp-action');
  const status = study.querySelector('.grasp-status');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const images = [...study.querySelectorAll('.grasp-image')];
  const contexts = images.map(image => image.getContext('2d'));
  const size = images[0].width;
  const imageData = contexts[0].createImageData(size, size);
  const sensorCanvases = images.map(() => {
    const sensor = document.createElement('canvas');
    sensor.width = sensor.height = size;
    return sensor;
  });
  const sensorContexts = sensorCanvases.map(sensor => sensor.getContext('2d'));
  const flowToggle = study.querySelector('.grasp-flow-toggle');
  let showFlow = true;
  let shape = 'sphere';
  let force = Number(slider.value);
  let gap = OPEN_GAP;
  let grasped = false;
  let motion = null;
  let dirty = true;
  let visible = true;
  let frame = null;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Interactive 3D gripper. Drag or use arrow keys to rotate the view. Use the shape, force, and grasp controls below to change contact.');
  host.appendChild(canvas);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#f0f2eb');
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 40);
  camera.position.set(3.2, 2.1, 4.6);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.38, 0);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableDamping = false;
  controls.minPolarAngle = 0.55;
  controls.maxPolarAngle = 1.65;
  controls.rotateSpeed = 0.7;
  controls.update();
  controls.saveState();
  controls.addEventListener('change', requestDraw);
  study.querySelector('.grasp-view-reset').addEventListener('click', () => controls.reset());
  canvas.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    if (event.key === 'ArrowLeft') spherical.theta -= 0.14;
    if (event.key === 'ArrowRight') spherical.theta += 0.14;
    if (event.key === 'ArrowUp') spherical.phi -= 0.12;
    if (event.key === 'ArrowDown') spherical.phi += 0.12;
    spherical.phi = THREE.MathUtils.clamp(spherical.phi, controls.minPolarAngle, controls.maxPolarAngle);
    camera.position.copy(new THREE.Vector3().setFromSpherical(spherical).add(controls.target));
    controls.update();
  });

  scene.add(new THREE.HemisphereLight('#fffaf0', '#718778', 2.8));
  const key = new THREE.DirectionalLight('#fff4e2', 3.2);
  key.position.set(-3, 6, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = key.shadow.camera.bottom = -3;
  key.shadow.camera.right = key.shadow.camera.top = 3;
  key.shadow.normalBias = 0.025;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#dbe8e0', 1.5);
  fill.position.set(3, 2, -3);
  scene.add(fill);

  const metal = new THREE.MeshStandardMaterial({ color: '#b8c5bc', roughness: 0.38, metalness: 0.35 });
  const housing = new THREE.MeshStandardMaterial({ color: '#405b4c', roughness: 0.52, metalness: 0.12 });
  const shell = new THREE.MeshStandardMaterial({ color: '#e0e4d9', roughness: 0.62 });
  const gel = new THREE.MeshStandardMaterial({ color: '#8fac9c', transparent: true, opacity: 0.45, roughness: 0.35, depthWrite: false });
  const objectMaterial = new THREE.MeshStandardMaterial({ color: '#c69a72', roughness: 0.5, metalness: 0.04 });
  const stageMaterial = new THREE.MeshStandardMaterial({ color: '#dce2d5', roughness: 0.9 });

  function mesh(geometry, material, parent, position) {
    const item = new THREE.Mesh(geometry, material);
    item.position.set(...position);
    item.castShadow = true;
    item.receiveShadow = true;
    parent.add(item);
    return item;
  }
  function box(w, h, d, material, parent, position) {
    return mesh(new RoundedBoxGeometry(w, h, d, 2, Math.min(0.035, w / 5, h / 5, d / 5)), material, parent, position);
  }
  mesh(new THREE.CylinderGeometry(1.58, 1.62, 0.08, 64), stageMaterial, scene, [0, -0.69, 0]);
  const pedestal = mesh(new THREE.CylinderGeometry(0.22, 0.25, 1, 32), housing, scene, [0, -0.57, 0]);
  box(2.3, 0.33, 0.65, shell, scene, [0, 1.16, 0]);
  box(2.13, 0.06, 0.69, housing, scene, [0, 1.055, 0]);
  mesh(new THREE.CylinderGeometry(0.23, 0.23, 0.27, 32), metal, scene, [0, 1.46, 0]);
  mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.12, 32), housing, scene, [0, 1.65, 0]);
  [-0.88, 0.88].forEach(x => {
    const screw = mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.012, 12), housing, scene, [x, 1.18, 0.333]);
    screw.rotation.x = Math.PI / 2;
  });

  const geometries = {
    sphere: new THREE.SphereGeometry(SHAPES.sphere.extent, 48, 32),
    cube: new THREE.BoxGeometry(0.84, 0.84, 0.84),
    cylinder: new THREE.CylinderGeometry(SHAPES.cylinder.extent, SHAPES.cylinder.extent, 0.98, 48)
  };
  const object = mesh(geometries.sphere, objectMaterial, scene, [0, 0, 0]);
  const pads = [-1, 1].map((side, index) => {
    const finger = new THREE.Group();
    scene.add(finger);
    box(0.38, 0.19, 0.59, metal, finger, [side * 0.12, 0.93, 0]);
    box(0.19, 1.35, 0.48, housing, finger, [side * 0.19, 0.26, 0]);
    box(0.18, 1.19, 1.18, shell, finger, [side * 0.19, 0, 0]);
    box(0.1, PAD_SIZE, PAD_SIZE, gel, finger, [side * 0.05, 0, 0]);
    const geometry = new THREE.PlaneGeometry(PAD_SIZE, PAD_SIZE, 32, 32);
    const vertices = geometry.attributes.position;
    const coordinates = [];
    for (let i = 0; i < vertices.count; i++) {
      coordinates.push([vertices.getY(i), vertices.getX(i)]);
      vertices.setXYZ(i, 0, vertices.getY(i), vertices.getX(i));
    }
    const texture = new THREE.CanvasTexture(sensorCanvases[index]);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.8, side: THREE.DoubleSide });
    const membrane = mesh(geometry, material, finger, [0, 0, 0]);
    membrane.castShadow = false;
    const labelCanvas = document.createElement('canvas');
    labelCanvas.width = labelCanvas.height = 64;
    const label = labelCanvas.getContext('2d');
    label.fillStyle = '#405b4c';
    label.beginPath(); label.arc(32, 32, 28, 0, Math.PI * 2); label.fill();
    label.fillStyle = '#fffdf6'; label.font = '500 36px sans-serif'; label.textAlign = 'center'; label.textBaseline = 'middle';
    label.fillText(index ? 'R' : 'L', 32, 34);
    const badge = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(labelCanvas) }));
    badge.position.set(side * 0.19, 0.68, 0.31); badge.scale.set(0.16, 0.16, 1); finger.add(badge);
    return { side, finger, geometry, vertices, coordinates, texture };
  });

  function drawTactile() {
    let activePixels = 0;
    for (let py = 0; py < size; py++) {
      const y = (0.5 - py / (size - 1)) * PAD_SIZE;
      for (let px = 0; px < size; px++) {
        const z = (px / (size - 1) - 0.5) * PAD_SIZE;
        const depth = contactDepth(shape, gap, y, z);
        const strength = Math.pow(Math.min(1, depth / MAX_INDENT), 0.65);
        if (depth > 0.00001) activePixels++;
        // Directional shading gives the imprint depth while color encodes load.
        const slope = depth > 0 ? (contactDepth(shape, gap, y + 0.012, z + 0.012) - contactDepth(shape, gap, y - 0.012, z - 0.012)) * 180 : 0;
        const i = (py * size + px) * 4;
        imageData.data[i] = 217 + strength * 3 + slope;
        imageData.data[i + 1] = 228 - strength * 83 + slope * 0.4;
        imageData.data[i + 2] = 215 - strength * 136 - slope * 0.3;
        imageData.data[i + 3] = 255;
      }
    }
    const ctx = contexts[0];
    ctx.putImageData(imageData, 0, 0);
    const markers = [];
    ctx.fillStyle = 'rgba(44, 75, 57, 0.48)';
    for (let py = 9; py < size; py += 14) {
      for (let px = 9; px < size; px += 14) {
        const y = (0.5 - py / (size - 1)) * PAD_SIZE;
        const z = (px / (size - 1) - 0.5) * PAD_SIZE;
        const displacement = markerDisplacement(shape, gap, y, z);
        // Magnify both the marker motion and its arrow by the same amount.
        const dx = displacement.x * 3;
        const dy = displacement.y * 3;
        markers.push({ px, py, dx, dy });
        ctx.beginPath(); ctx.arc(px + dx, py + dy, 0.85, 0, Math.PI * 2); ctx.fill();
      }
    }
    // The gripper's gel shows markers; vector annotations belong only in readouts.
    sensorContexts[0].drawImage(images[0], 0, 0);
    sensorContexts[1].save();
    sensorContexts[1].translate(size, 0); sensorContexts[1].scale(-1, 1);
    sensorContexts[1].drawImage(images[0], 0, 0);
    sensorContexts[1].restore();
    let arrowCount = 0;
    if (showFlow) {
      ctx.strokeStyle = '#294d3a';
      ctx.fillStyle = '#294d3a';
      ctx.lineWidth = 1.3;
      ctx.lineCap = 'round';
      markers.forEach(({ px, py, dx, dy }) => {
        const length = Math.hypot(dx, dy);
        if (length < 1.5) return;
        arrowCount++;
        const ux = dx / length, uy = dy / length;
        const tipX = px + dx, tipY = py + dy;
        const head = Math.min(3.6, length * 0.45);
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tipX, tipY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - ux * head - uy * head * 0.6, tipY - uy * head + ux * head * 0.6);
        ctx.lineTo(tipX - ux * head + uy * head * 0.6, tipY - uy * head - ux * head * 0.6);
        ctx.closePath(); ctx.fill();
      });
    }
    study.dataset.flowArrows = String(arrowCount);
    contexts[1].save();
    contexts[1].translate(size, 0); contexts[1].scale(-1, 1);
    contexts[1].drawImage(images[0], 0, 0);
    contexts[1].restore();
    pads.forEach(pad => { pad.texture.needsUpdate = true; });
    study.dataset.contact = String(activePixels > 0);
    images.forEach((image, index) => {
      const description = activePixels ? SHAPES[shape].description + ', ' + force + '% target force' : 'no contact';
      image.setAttribute('aria-label', (index ? 'Right' : 'Left') + ' tactile image: ' + description + (arrowCount ? ', arrows show marker displacement from unloaded positions, magnified three times' : ''));
    });
  }

  function updateContact() {
    object.geometry = geometries[shape];
    const supportHeight = SHAPES[shape].bottom + 0.65;
    pedestal.scale.y = supportHeight;
    pedestal.position.y = -0.65 + supportHeight / 2;
    pads.forEach(pad => {
      pad.finger.position.x = pad.side * gap;
      pad.coordinates.forEach(([y, z], i) => {
        // The sensing surface conforms to the same object surface that produces the image.
        pad.vertices.setX(i, pad.side * (contactDepth(shape, gap, y, z) + 0.001));
      });
      pad.vertices.needsUpdate = true;
      pad.geometry.computeVertexNormals();
    });
    drawTactile();
    const contact = contactDepth(shape, gap, 0, 0) > 0.00001;
    const state = motion ? (grasped ? 'closing' : 'opening') : (grasped ? 'grasped' : 'open');
    study.dataset.state = state;
    study.dataset.shape = shape;
    study.dataset.gap = gap.toFixed(5);
    const text = motion ? (grasped ? 'Closing' : 'Opening') + (contact ? ' · contact' : ' · no contact') : contact ? 'Contact · ' + shape : grasped ? 'Touching · no load' : 'Open · no contact';
    if (status.textContent !== text) status.textContent = text;
  }

  function requestDraw() {
    if (frame === null && visible && !document.hidden) frame = requestAnimationFrame(draw);
  }
  function draw(time) {
    frame = null;
    if (!visible || document.hidden) return;
    if (motion) {
      const t = reducedMotion.matches ? 1 : Math.min(1, (time - motion.start) / 850);
      gap = THREE.MathUtils.lerp(motion.from, motion.to, t * (2 - t));
      if (t >= 1) motion = null;
      dirty = true;
    }
    if (dirty) { updateContact(); dirty = false; }
    renderer.render(scene, camera);
    if (motion) requestDraw();
  }
  function moveTo(target) {
    motion = { from: gap, to: target, start: performance.now() };
    requestDraw();
  }
  action.addEventListener('click', () => {
    grasped = !grasped;
    action.innerHTML = grasped ? 'Release object <span aria-hidden="true">↔</span>' : 'Grasp object <span aria-hidden="true">→</span>';
    moveTo(grasped ? gripGap(shape, force) : OPEN_GAP);
  });
  study.querySelectorAll('[data-shape]').forEach(button => {
    button.addEventListener('click', () => {
      if (shape === button.dataset.shape) return;
      // Reopen before replacing geometry; a new object never inherits contact.
      shape = button.dataset.shape;
      grasped = false; motion = null; gap = OPEN_GAP;
      action.innerHTML = 'Grasp object <span aria-hidden="true">→</span>';
      study.querySelectorAll('[data-shape]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      dirty = true;
      requestDraw();
    });
  });
  slider.addEventListener('input', () => {
    force = Number(slider.value);
    output.value = force + '%';
    if (grasped) moveTo(gripGap(shape, force));
  });
  flowToggle.addEventListener('click', () => {
    showFlow = !showFlow;
    flowToggle.setAttribute('aria-pressed', String(showFlow));
    dirty = true;
    requestDraw();
  });

  const resize = new ResizeObserver(() => {
    const { width, height } = host.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    // Keep the open jaws in frame even in a narrow mobile viewport.
    camera.zoom = Math.min(1, camera.aspect / 1.05);
    camera.updateProjectionMatrix();
    requestDraw();
  });
  resize.observe(host);
  document.addEventListener('visibilitychange', requestDraw);
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    visible = false;
    study.classList.remove('is-ready');
    study.querySelector('.grasp-interface').hidden = true;
    study.querySelector('.grasp-readouts').hidden = true;
    study.querySelector('.grasp-view-reset').hidden = true;
    study.querySelector('.grasp-orbit-hint').hidden = true;
    study.querySelector('.grasp-fallback-note').textContent = 'The 3D view was interrupted. Reload the page to try again.';
  });
  study.querySelector('.grasp-interface').hidden = false;
  study.querySelector('.grasp-readouts').hidden = false;
  study.querySelector('.grasp-view-reset').hidden = false;
  study.querySelector('.grasp-orbit-hint').hidden = false;
  study.classList.add('is-ready');
  updateContact();
  requestDraw();
  return value => { visible = value; if (value) requestDraw(); };
}
