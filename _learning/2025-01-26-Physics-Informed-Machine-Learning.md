---
title: 'Neural ODEs'
date: 2025-01-26
permalink: /neural-odes/
tags:
  - Physics
  - Differentiable Equations
  - Machine Learning
  - Neural ODEs
  - NODEs
---

**Authors:**  Wong Lik Hang Kenny

Reference: 

- [Neural ODEs (NODEs) \[Physics Informed Machine Learning\] YouTube Video](https://www.youtube.com/watch?v=nJphsM4obOk&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=19)
- [Original paper](https://arxiv.org/abs/1806.07366)
- ChatGPT-4o (for polishing and content structuring)

# **Neural Ordinary Differential Equations (Neural ODEs)**

Neural Ordinary Differential Equations (Neural ODEs) are a powerful class of machine learning models that unify discrete, layer-based neural architectures (like residual networks) with continuous dynamical systems. They were introduced in the 2018 NeurIPS paper by Chen, Rubanova, Bettencourt, and Duvenaud, titled **“Neural Ordinary Differential Equations.”**

This blog post provides a gentle yet detailed introduction to Neural ODEs, explains why they generalize certain deep network architectures, how they can be trained, and illustrates their exciting properties. We will conclude with a discussion about how Neural ODEs can be applied in **robotics** and **physics**.

---

## **Table of Contents**
- [**Neural Ordinary Differential Equations (Neural ODEs)**](#neural-ordinary-differential-equations-neural-odes)
  - [**Table of Contents**](#table-of-contents)
  - [1. Motivation: From Residual Networks to Continuous Depth](#1-motivation-from-residual-networks-to-continuous-depth)
  - [2. What Is an ODE and Why Make It Neural?](#2-what-is-an-ode-and-why-make-it-neural)
  - [3. Defining a Neural ODE](#3-defining-a-neural-ode)
  - [4. Training Neural ODEs via the Adjoint Method](#4-training-neural-odes-via-the-adjoint-method)
  - [5. Advantages and Properties of Neural ODEs](#5-advantages-and-properties-of-neural-odes)
    - [**(a) Memory Efficiency**](#a-memory-efficiency)
    - [**(b) Adaptive Computation**](#b-adaptive-computation)
    - [**(c) Continuous Normalizing Flows**](#c-continuous-normalizing-flows)
    - [**(d) Continuous-Time Modeling of Data**](#d-continuous-time-modeling-of-data)
  - [6. Applications in Robotics](#6-applications-in-robotics)
  - [7. Applications in Physics](#7-applications-in-physics)
  - [8. Concluding Remarks](#8-concluding-remarks)

---

<a name="motivation"></a>
## 1. Motivation: From Residual Networks to Continuous Depth

To appreciate **Neural ODEs**, let’s start with an important building block in modern deep learning: the **residual block**. A **ResNet** layer updates its hidden state $$\mathbf{h}_t$$ as follows:

$$
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta_t),
$$

where $$f$$ is a learnable function (often a neural network itself), and $$\theta_t$$ are its parameters. 

- **Interpretation as Euler’s method**: If you squint at this update, you’ll notice it looks like a single “step” of the **Euler method** for solving a differential equation $$\frac{d\mathbf{h}}{dt} = f(\mathbf{h}, t, \theta)$$.  
- As we **stack more layers** (or equivalently, take smaller and smaller steps), this residual update approaches a **continuous** transformation of $$\mathbf{h}$$. 

**Neural ODEs** extend this idea by directly modeling **$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t)$$** and using **ODE solvers** instead of fixed discrete layers.

---

<a name="what-is-an-ode"></a>
## 2. What Is an ODE and Why Make It Neural?

An **Ordinary Differential Equation (ODE)** describes how a variable $$\mathbf{x}(t)$$ changes over continuous time $$t$$. For example:

$$
\frac{d\mathbf{x}(t)}{dt} \;=\; f\bigl(\mathbf{x}(t), t\bigr).
$$

- If you specify an **initial condition** $$\mathbf{x}(0)$$, the ODE solutions are the unique trajectories $$\mathbf{x}(t)$$ that satisfy this equation.
- Solving an ODE often means using **numerical ODE solvers** (e.g., Runge-Kutta methods, Adams methods) to step forward in time.

**Making it Neural** means letting the **function $$f$$** be parameterized by a neural network—e.g.,

$$
f(\mathbf{x}(t), t; \theta) \;=\; \text{NeuralNetwork}(\mathbf{x}(t), t; \theta).
$$

We then rely on standard ODE solvers to **integrate** this neural network over time.

---

<a name="defining-a-neural-ode"></a>
## 3. Defining a Neural ODE

Formally, a Neural ODE sets:

$$
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta).
$$

Starting from an **input** $$\mathbf{h}(0)$$, we obtain an **output** $$\mathbf{h}(T)$$ for some final time $$T$$ by solving this initial value problem (IVP):

$$
\mathbf{h}(T) = \mathbf{h}(0) + \int_{0}^{T} f\bigl(\mathbf{h}(t), t, \theta \bigr)\, dt.
$$

In practice, we never compute that integral exactly by hand. Instead, we hand $$\mathbf{h}(0)$$ and $$f$$ to a **black-box ODE solver**, which:
1. Dynamically decides time steps.
2. Evaluates $$f$$ at necessary points.
3. Produces a numerical approximation to $$\mathbf{h}(T)$$.

---

<a name="training-neural-odes"></a>
## 4. Training Neural ODEs via the Adjoint Method

One key challenge is how to **train** the parameters $$\theta$$. Traditional backpropagation would require you to store every intermediate step in memory, which can be expensive or even infeasible for large or adaptive solvers.

**The adjoint sensitivity method** offers a clever solution:
- Instead of storing every intermediate state, we run the ODE solver **forward** to obtain $$\mathbf{h}(T)$$.
- We compute the **loss** $$L(\mathbf{h}(T))$$.
- Then, we introduce an **adjoint variable** $$\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{h}(t)}$$, which we solve **backwards in time** along with $$\mathbf{h}(t)$$.

The reverse-time ODE for $$\mathbf{a}(t)$$ involves Jacobian-vector products of $$f$$. Crucially, automatic differentiation (as used in modern deep learning) can compute these products efficiently. 

This approach:
1. Keeps memory cost **constant** w.r.t. the number of forward solver steps.
2. Allows the user to treat the ODE solver as a **black box** while still computing gradients for training.

---

<a name="advantages"></a>
## 5. Advantages and Properties of Neural ODEs

### <a name="memory-efficiency"></a>**(a) Memory Efficiency**

Because the **adjoint method** does not store intermediate activations for each ODE solver step, Neural ODEs scale to deeper/continuous architectures with **constant memory cost**. 

- In classical feed-forward or residual networks, memory grows with the number of layers (or steps).
- In a Neural ODE, we effectively have *infinitely many layers* in principle, but do not pay a huge memory cost for that.

### <a name="adaptive-computation"></a>**(b) Adaptive Computation**

Traditional discrete networks use a fixed number of layers. By contrast, ODE solvers:
- Adapt the **step size** to keep numerical error below a desired tolerance.
- Potentially reduce computation by taking fewer steps in **easy** regions.
- Provide finer resolution where the solution changes quickly.

### <a name="continuous-normalizing-flows"></a>**(c) Continuous Normalizing Flows**

A powerful application is **Continuous Normalizing Flows (CNFs)** for generative modeling. Standard normalizing flows rely on Jacobian determinants, which can be expensive for high-dimensional transformations. In **CNFs**, the log-probability evolves with a **trace** of the Jacobian, often simpler and more scalable.

### <a name="continuous-time-modeling"></a>**(d) Continuous-Time Modeling of Data**

Many real-world processes (especially physical or robotic systems) are inherently **continuous** in time. Neural ODEs let you:
- Train on data sampled **irregularly** in time.
- Generate predictions for **any** time, even times not in the training set.
- Leverage advanced integrators to preserve domain-specific structure (e.g., symplectic integrators for Hamiltonian systems).

---

<a name="robotics-applications"></a>
## 6. Applications in Robotics

In robotics, the dynamics of manipulators, drones, or walking robots often follow continuous-time physics:

1. **Learning Robot Dynamics**:  
   - Many robotics tasks require precise system identification—figuring out how states like position, velocity, and torque evolve over time.  
   - With Neural ODEs, you can learn these continuous dynamics from partial or noisy measurements.  

2. **Trajectory Optimization**:  
   - If you have a trained Neural ODE representing robot dynamics, you can integrate it to plan or optimize trajectories.  
   - This is especially useful when you have data sampled at *uneven intervals* or the robot’s movements are fast in some phases and slow in others.

3. **Energy or Constraint Preservation**:  
   - By carefully choosing specialized ODE solvers (e.g., symplectic integrators) within a Neural ODE framework, you can preserve known physical constraints such as conservation of energy.  
   - This ensures more stable and realistic robotic control.

**In practice**: You can embed a Neural ODE block inside a larger control architecture, letting it serve as an **adaptive and data-driven model** of your system’s continuous motion.

---

<a name="physics-applications"></a>
## 7. Applications in Physics

Many physical processes—like planetary motion, fluid flows, chemical kinetics—are naturally described by differential equations:

1. **Data-Driven Discovery of Dynamics**:  
   - Traditional PDE or ODE derivations can be very complex. Neural ODEs let you approximate the right-hand side of physical laws **directly from data**.  

2. **Hamiltonian/Lagrangian Neural Networks**:  
   - Extensions of Neural ODEs, such as **Hamiltonian Neural Networks** or **Lagrangian Neural Networks**, incorporate energy-conserving properties or variational principles into the architecture.  
   - This can be helpful in modeling and simulating physical systems **long-term** without numerical drift in energy.

3. **Irregular Sampling**:  
   - Physical measurements are often taken at **uneven intervals** (due to sensor scheduling, experimental constraints, etc.). Neural ODEs are particularly convenient in this setting, as the solver adaptively handles time steps.

4. **Predictive Modeling and Forecasting**:  
   - Once trained, a Neural ODE can predict future states (e.g., pressure, temperature, or mechanical stress) by simply integrating forward in time.  
   - Similarly, you can model long-term behaviors more stably than with discrete updates, if you choose appropriate integrators.

---

<a name="conclusion"></a>
## 8. Concluding Remarks

**Neural ODEs** mark a conceptual shift from thinking of neural networks as layered, discrete transformations to viewing them as **continuously evolving dynamical systems**. This perspective:
- **Bridges** machine learning with classical ODE-based modeling.
- **Improves** memory usage and allows for **adaptive** evaluation.
- Offers new avenues in **generative modeling**, **robotics**, and **physics**.

When you need to handle **irregularly sampled data**, embed domain-specific constraints, or require memory-efficient training of very deep models, Neural ODEs can be a compelling solution.  

Whether you’re building a **manipulator control system** or studying **planetary orbits**, Neural ODEs provide a flexible, elegant approach to learning continuous-time dynamics from data. It is an exciting technique that continues to inspire new research and innovative applications in the broader landscape of **physics-informed machine learning**.

---