# Navier-Stokes Regularity

```
If the fluid blows up, the topology collapses first.
If it doesn't, the topology tells you why.
```

## The Problem

The 3D incompressible Navier-Stokes equations describe fluid motion. The Millennium Problem asks: given smooth initial data, does the solution remain smooth for all time, or can it develop a singularity (infinite velocity or vorticity) in finite time? Nobody knows. The equations have been around since the 1840s. The regularity question has been open since Leray's 1934 work.

The difficulty is analytical — the nonlinear term resists every closure technique. Numerical simulations suggest the flow stays regular, but simulations can't prove it because they discretize away exactly the small-scale behavior where blow-up would occur.

## The ATFT Translation

**Point cloud:** Take a snapshot of the vorticity field omega(x, t) from a 3D fluid simulation. Extract vortex line positions — the skeleton of the flow's rotational structure. Each vortex segment becomes a point in R^3 (position) or R^6 (position + local vorticity direction). The point cloud evolves with the fluid.

**Control parameter:** Reynolds number Re (or equivalently, time t in a decaying flow). As Re increases, the vortex structures get more tangled and smaller-scale. If blow-up happens, it happens because vortex lines collapse to a point faster than viscosity can smooth them.

**Sheaf:** Attach R^3-valued fibers encoding local vorticity orientation. The transport map on the Rips complex carries the vorticity alignment structure between neighboring vortex segments.

**Detection target:** The onset scale epsilon*(t) tracks the **vortex coherence length** — the scale at which the persistent homology first detects connected structure in the vortex cloud. This is a topological proxy for the smallest active scale in the flow.

Here's the key insight: **if a singularity develops, epsilon* goes to zero.** The point cloud collapses. All topology concentrates at infinitesimal scale. The persistence diagram degenerates. Conversely, **if regularity holds, epsilon* stays bounded away from zero** at all times. The topology never collapses. There's always a minimum coherence length.

The Navier-Stokes regularity question, in ATFT language: **does the topological evolution curve epsilon*(t) have a waypoint at epsilon = 0?**

## Experimental Protocol

1. **Taylor-Green vortex** (simplest test): Initialize the classic TG vortex on a periodic domain. Run DNS via pseudospectral method (dealiased, 256^3 or 512^3 grid). Extract vortex lines at each time step. Track epsilon*(t) as the flow evolves through its energy cascade.

2. **Kida-Pelz flow** (harder test): Known to develop intense vorticity growth. Track whether epsilon*(t) approaches zero or levels off.

3. **Synthetic blow-up candidates**: Construct point clouds that mimic hypothetical blow-up scenarios (self-similar collapse of vortex rings). Check that epsilon* does go to zero — this validates that the instrument would detect blow-up if it existed.

**Hardware:** i9-9900K + RTX 5070. DNS at 256^3 is feasible on GPU with PyTorch FFT. 512^3 is tight but possible with careful memory management. Each snapshot produces O(10^4) vortex points — well within the persistent homology pipeline's capacity.

## What Success Looks Like

- **Regularity signal:** epsilon*(t) for Taylor-Green vortex remains bounded below by some epsilon_min > 0, even as enstrophy peaks. The topology stays coherent. The Gini trajectory stays structured — the flow is complex but organized.

- **Blow-up signal (synthetic):** epsilon*(t) for designed collapse scenarios hits zero in finite time, confirming the instrument detects the right thing.

- **Diagnostic power:** The epsilon*(t) curve provides earlier warning of incipient blow-up than traditional diagnostics (max vorticity, enstrophy). The topology sees the collapse before the PDE does.

## What's Speculative

Everything about blow-up is speculative — that's the problem. We can't prove regularity by tracking epsilon*(t) in a finite simulation, because blow-up might happen at times or Reynolds numbers beyond what we simulate. What we CAN do is establish that the topological evolution curve is a faithful diagnostic, show it tracks known phenomena correctly, and identify what a blow-up signature would look like if one existed.

The honest statement: this won't solve Navier-Stokes. It will build a new diagnostic instrument for the flow topology that encodes the regularity question in a measurable, scale-aware observable. If someone eventually finds blow-up initial data, the ATFT operator should see it first.

---

*The fluid doesn't care about your equations. It cares about its vortex lines. The topology reads the lines.*
