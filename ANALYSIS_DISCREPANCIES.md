# Analysis: Discrepancies Between Python and Go Implementations

**Date:** 2026-01-19
**Files compared:**
- `r4bp-Auto-Solver.py` (Python)
- `rewrite.go` (Go)

## Executive Summary

The Go rewrite produces different results from the Python original due to a **bug in the Python code** that inadvertently creates favorable trajectory conditions. The Python code achieves lunar capture while Go does not, despite Go having the mathematically correct implementation.

| Metric | Python | Go |
|--------|--------|-----|
| Phase 1 end time | 11,294,813 s | 11,310,277 s |
| Phase 1 end x-position | 26,166 km | 34,472 km |
| Phase 2 duration | 15.9 million s | 3.7 million s |
| Total time | 319.02 days | 177.36 days |
| Capture success | ✅ Yes | ❌ No |
| SOI entry | ✅ Yes | ❌ No |

---

## 🔴 Primary Issue: Python Bug in Jacobi Event Function

### Location
`r4bp-Auto-Solver.py` lines 424-440

### Description
The `jacobiC_local` event function uses the **initial sun angle** directly instead of computing the time-evolved position.

**Buggy code (Python):**
```python
def jacobiC_local(t, state, phiS):  # phiS receives phiS0_rad (initial angle)
    ...
    # BUG: Uses initial sun angle instead of time-evolved angle
    xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
    ...
```

**Correct approach (used in rates function and in Go):**
```python
phiS = phiS0_rad + (nS - W) * t  # Time-evolved angle
xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
```

### Impact
Over 11 million seconds of phase 1, the sun rotates approximately **1596 degrees** (~4.4 revolutions). This causes significantly different Jacobi constant values:

| Implementation | Sun Angle Used | Jacobi Value | Distance from Threshold |
|----------------|----------------|--------------|------------------------|
| Python (buggy) | 30° (initial) | -1.63903 | 0.00003 |
| Go (correct) | 234° (evolved) | -1.64057 | 0.00157 |

The Jacobi event triggers at different spacecraft states, causing an **8,300 km difference** in x-position at phase 1 end.

### Root Cause
In `solve_ivp`, the `args=(phiS0_rad,)` parameter is passed to both the rates function AND event functions. The rates function correctly computes the time-evolved sun position internally, but the event function uses the passed value directly.

---

## 🟡 Secondary Issue: Sun Position Discontinuity Between Phases

### Location
- `r4bp-Auto-Solver.py` line 467
- `rewrite.go` line 509

### Description
Both implementations update `phiS0` between phases using:
```
phiS0_new = phiS0 - (nS - W) * t_end_of_phase
```

This formula creates a **discontinuity** in the sun position:

| Time Point | Expected Sun Position | Actual (with formula) |
|------------|----------------------|----------------------|
| End of Phase 1 | -1566° | -1566° |
| Start of Phase 2 | -1566° (continuous) | **30°** (reset!) |

The sun "jumps back" approximately 1596 degrees at each phase boundary.

### Impact
Both implementations have this bug, so it doesn't explain the discrepancy between them. However, it is physically incorrect and may affect trajectory accuracy.

---

## 🟡 Secondary Issue: Integration Step Size Differences

### Description
Python uses different `max_step` values for each phase, while Go uses a constant value:

| Phase | Python max_step | Go MaxStep |
|-------|-----------------|------------|
| 1 (Thrust) | 450 s | 450 s |
| 2 (Coast) | **100 s** | 450 s |
| 3 (Brake) | **100 s** | 450 s |

### Impact
Go uses 4.5× larger integration steps during phases 2 and 3. This could reduce accuracy during the sensitive lunar approach phase, though adaptive step control should mitigate this.

---

## 🟡 Secondary Issue: L1 Event Function Behavior Difference

### Location
- `r4bp-Auto-Solver.py` lines 114-130
- `rewrite.go` lines 223-235

### Description
The L1 event functions handle out-of-window cases differently:

**Python (preserves sign):**
```python
if abs(y_val) <= Y_WINDOW_L1:
    return s  # Distance from L1
margin = (abs(y_val) - Y_WINDOW_L1) + eps
return (1.0 if s >= 0.0 else -1.0) * margin  # Sign based on x position
```

**Go (always positive outside window):**
```go
if absY <= yWindow {
    return dx  // Distance from L1
}
return absY - yWindow + math.Abs(dx) + 1  // Always positive
```

### Test Results
| Spacecraft Position | Python Returns | Go Returns |
|--------------------|----------------|------------|
| Inside window, before L1 | -17039 | -17039 |
| Inside window, past L1 | +2961 | +2961 |
| **Outside window, before L1** | **-5000** | **+22040** |
| Outside window, past L1 | +5000 | +7962 |

### Impact
When spacecraft is outside y-window and before L1:
- Python returns **negative** (maintains sign continuity)
- Go returns **positive** (could cause spurious sign change detection)

This could cause Go to incorrectly detect an L1 "event" when the spacecraft enters the y-window.

---

## 🟡 Minor Issue: Error Norm Calculation

### Location
`rewrite.go` line 358-360

### Description
The RK45 error norm calculation divides by 7 (for x, y, z, vx, vy, vz, mass components), but z and vz are always 0 in this 2D problem:

```go
errNorm := math.Sqrt((md3.Norm2(errPos)/(scalePos*scalePos) +
    md3.Norm2(errVel)/(scaleVel*scaleVel) +
    errM*errM/(scaleM*scaleM)) / 7)  // Should be 5 for 2D problem?
```

### Impact
This slightly underestimates the error norm, potentially allowing larger-than-optimal step sizes.

---

## Constants Verification

All physical constants match between implementations:

| Constant | Value | Match |
|----------|-------|-------|
| W (angular velocity) | 2.6653136671e-06 rad/s | ✅ |
| nS (sun mean motion) | 1.9912641816e-07 rad/s | ✅ |
| mu (combined grav param) | 403503.02 km³/s² | ✅ |
| muS (sun grav param) | 1.3274983800e+11 km³/s² | ✅ |
| x1 (Earth position) | -4670.66 km | ✅ |
| x2 (Moon position) | 379729.34 km | ✅ |
| L1x (L1 x-coordinate) | 317039.34 km | ✅ |

## Initial State Verification

Initial conditions match exactly:

| Parameter | Python | Go | Match |
|-----------|--------|-----|-------|
| Position | (14004.05, -39152.34) km | (14004.05, -39152.34) km | ✅ |
| Velocity | (2.631685, 1.255249) km/s | (2.631685, 1.255249) km/s | ✅ |
| Mass | 12.0 kg | 12.0 kg | ✅ |

---

## Recommendations

### Option 1: Make Go Replicate Python's Behavior
To achieve identical results, modify Go's `EventJacobi` to use the initial sun angle:

```go
func EventJacobi(threshold float64) EventFunc {
    return func(t float64, s State, phiS0 float64) float64 {
        // Use phiS0 directly (replicating Python bug)
        return JacobiConstantBuggy(s, phiS0) - threshold
    }
}
```

### Option 2: Fix Python's Bug
Modify `jacobiC_local` to compute time-evolved sun position:

```python
def jacobiC_local(t, state, phiS0):
    ...
    phiS = phiS0 + (nS - W) * t  # Compute current sun angle
    xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
    ...
```

### Option 3: Fix Both Issues
1. Fix the Jacobi event calculation in both implementations
2. Fix the phiS0 update formula to maintain continuity:
   ```
   # Don't update phiS0 between phases - keep using original
   # OR use: phiS0_new = phiS0 (no change)
   ```

### Additional Fixes for Go
1. Update max_step for phases 2&3 to match Python (100s)
2. Fix L1 event function to preserve sign outside y-window
3. Consider using 5 instead of 7 in error norm calculation

---

## Conclusion

The primary cause of discrepancy is **Python's buggy Jacobi event function** that uses the initial sun angle instead of the time-evolved position. This bug happens to produce favorable trajectory conditions for the test parameters (phi=295.5°, phiS0=30°), allowing successful lunar capture.

Go's mathematically correct implementation finds a different solution that does not achieve capture with these parameters. To replicate Python's results, Go would need to intentionally reproduce the bug.
