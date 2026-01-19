package main

import (
	"fmt"
	"math"
	"time"

	"github.com/soypat/geometry/md3"
)

// Physical constants for Earth-Moon-Sun system.
const (
	days   = 24 * 3600    // [s/day]
	bigG   = 6.6742e-20   // [km³/kg/s²]
	r12    = 384400.0     // [km] Earth-Moon distance
	m1     = 5.974e24     // [kg] Earth mass
	m2     = 7.348e22     // [kg] Moon mass
	mu1    = 398600.0     // [km³/s²] Earth gravitational parameter
	mu2    = 4903.02      // [km³/s²] Moon gravitational parameter
	mS     = 1.989e30     // [kg] Sun mass
	rB2S   = 149597870.7  // [km] Barycenter to Sun distance
	rMoon  = 1737.0       // [km] Moon radius
	rEarth = 6378.0       // [km] Earth radius
	L1dist = 321710.0     // [km] L1 distance from Earth center
	C1     = -1.676       // Jacobi constant for braking phase
)

// Derived constants (computed at init).
var (
	mu  = mu1 + mu2            // [km³/s²] Combined gravitational parameter
	pi1 = m1 / (m1 + m2)       // Earth mass fraction
	pi2 = m2 / (m1 + m2)       // Moon mass fraction
	W   = math.Sqrt(mu / (r12 * r12 * r12)) // [rad/s] Angular velocity of rotating frame
	x1  = -pi2 * r12           // [km] Earth x-position in rotating frame
	x2  = pi1 * r12            // [km] Moon x-position in rotating frame
	muS = bigG * mS            // [km³/s²] Sun gravitational parameter
	nS  = math.Sqrt(muS / (rB2S * rB2S * rB2S)) // [rad/s] Sun mean motion
	L1x = x1 + L1dist          // [km] L1 x-coordinate
)

// State represents spacecraft state in the rotating Earth-Moon frame.
type State struct {
	Pos  md3.Vec // [km]
	Vel  md3.Vec // [km/s]
	Mass float64 // [kg]
}

// Thruster defines propulsion parameters.
type Thruster struct {
	Thrust float64 // [kN] Total thrust
	Isp    float64 // [s] Specific impulse
}

// MassRate returns mass flow rate [kg/s] (negative, mass decreases).
func (th Thruster) MassRate() float64 {
	const g0 = 9.807e-3 // [km/s²]
	return -th.Thrust / (g0 * th.Isp)
}

// Trajectory holds initial conditions and computes transfer trajectory.
type Trajectory struct {
	D0    float64 // [km] Altitude above Earth
	Phi   float64 // [rad] Initial angle from Earth center
	Gamma float64 // [rad] Flight path angle
	PhiS0 float64 // [rad] Initial Sun angle in rotating frame
	M0    float64 // [kg] Initial mass

	Thruster  Thruster
	JacobiThr float64 // Jacobi threshold for phase 1 termination
	Tol       float64 // Integration tolerance
	MaxStep   float64 // [s] Maximum integration step
}

// Result holds trajectory simulation output.
type Result struct {
	TotalTime float64 // [s]
	FinalMass float64 // [kg]
	ReachedL1 bool
	Captured  bool
	Collided  bool
	InSOI     bool // Entered Moon's sphere of influence
}

// InitialState computes the initial state from trajectory parameters.
func (tr *Trajectory) InitialState() State {
	r0 := rEarth + tr.D0
	v0 := math.Sqrt(mu1/r0) - W*r0 // Circular velocity minus frame rotation

	sinPhi, cosPhi := math.Sincos(tr.Phi)
	sinGam, cosGam := math.Sincos(tr.Gamma)

	return State{
		Pos: md3.Vec{
			X: r0*cosPhi + x1,
			Y: r0 * sinPhi,
			Z: 0,
		},
		Vel: md3.Vec{
			X: v0 * (sinGam*cosPhi - cosGam*sinPhi),
			Y: v0 * (sinGam*sinPhi + cosGam*cosPhi),
			Z: 0,
		},
		Mass: tr.M0,
	}
}

// RatesFunc computes state derivatives: dPos/dt, dVel/dt, dm/dt.
type RatesFunc func(t float64, s State, phiS0 float64) (dPos, dVel md3.Vec, dm float64)

// EventFunc returns a value that crosses zero at an event.
type EventFunc func(t float64, s State, phiS0 float64) float64

// SunPos returns Sun position in rotating frame at time t.
func SunPos(t, phiS0 float64) md3.Vec {
	phi := phiS0 + (nS-W)*t
	sin, cos := math.Sincos(phi)
	return md3.Vec{X: rB2S * cos, Y: rB2S * sin, Z: 0}
}

// Accel computes gravitational acceleration from Earth, Moon, and Sun.
// Does not include thrust or Coriolis/centrifugal terms.
func Accel(pos md3.Vec, t, phiS0 float64) md3.Vec {
	// Vectors to Earth, Moon
	rToEarth := md3.Sub(pos, md3.Vec{X: x1, Y: 0, Z: 0})
	rToMoon := md3.Sub(pos, md3.Vec{X: x2, Y: 0, Z: 0})
	d1 := md3.Norm(rToEarth)
	d2 := md3.Norm(rToMoon)

	// Sun position and distance
	sunPos := SunPos(t, phiS0)
	rToSun := md3.Sub(pos, sunPos)
	dS := md3.Norm(rToSun)

	// Gravitational accelerations
	d1_3 := d1 * d1 * d1
	d2_3 := d2 * d2 * d2
	dS_3 := dS * dS * dS
	rB2S_3 := rB2S * rB2S * rB2S

	aEarth := md3.Scale(-mu1/d1_3, rToEarth)
	aMoon := md3.Scale(-mu2/d2_3, rToMoon)
	aSunDirect := md3.Scale(-muS/dS_3, rToSun)
	aSunIndirect := md3.Scale(-muS/rB2S_3, sunPos)

	return md3.Add(md3.Add(aEarth, aMoon), md3.Add(aSunDirect, aSunIndirect))
}

// RatesThrust returns derivatives with thrust along velocity direction.
func RatesThrust(th Thruster) RatesFunc {
	return func(t float64, s State, phiS0 float64) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel

		aGrav := Accel(s.Pos, t, phiS0)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X, Z: 0}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y, Z: 0}

		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(th.Thrust/(s.Mass*v), s.Vel)

		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return
	}
}

// RatesCoast returns derivatives with no thrust.
func RatesCoast(t float64, s State, phiS0 float64) (dPos, dVel md3.Vec, dm float64) {
	dPos = s.Vel

	aGrav := Accel(s.Pos, t, phiS0)
	aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X, Z: 0}
	aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y, Z: 0}

	dVel = md3.Add(md3.Add(aGrav, aCoriolis), aCentrifugal)
	return
}

// RatesBrake returns derivatives with thrust opposing velocity.
func RatesBrake(th Thruster) RatesFunc {
	return func(t float64, s State, phiS0 float64) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel

		aGrav := Accel(s.Pos, t, phiS0)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X, Z: 0}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y, Z: 0}

		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(-th.Thrust/(s.Mass*v), s.Vel) // Negative = braking

		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return
	}
}

// JacobiConstant computes the Jacobi integral for 4-body problem.
func JacobiConstant(s State, t, phiS0 float64) float64 {
	v2 := md3.Norm2(s.Vel)
	pos := s.Pos

	d1 := math.Hypot(pos.X-x1, pos.Y)
	d2 := math.Hypot(pos.X-x2, pos.Y)
	sunPos := SunPos(t, phiS0)
	dS := md3.Norm(md3.Sub(pos, sunPos))

	U_em := 0.5*v2 - 0.5*W*W*(pos.X*pos.X+pos.Y*pos.Y) - mu1/d1 - mu2/d2
	U_sun_direct := -muS/dS + muS/rB2S
	rB2S_3 := rB2S * rB2S * rB2S
	U_sun_indirect := muS * (pos.X*sunPos.X + pos.Y*sunPos.Y) / rB2S_3

	return U_em + U_sun_direct + U_sun_indirect
}

// EventJacobi returns event function for Jacobi threshold crossing.
func EventJacobi(threshold float64) EventFunc {
	return func(t float64, s State, phiS0 float64) float64 {
		return JacobiConstant(s, t, phiS0) - threshold
	}
}

// EventL1 returns event function for L1 crossing within y-window.
// Matches Python lagranian1 logic exactly.
func EventL1(yWindow float64) EventFunc {
	return func(t float64, s State, phiS0 float64) float64 {
		dx := s.Pos.X - L1x
		absY := math.Abs(s.Pos.Y)
		if absY <= yWindow {
			return dx
		}
		// Outside window: maintain sign of dx but with offset
		// This is continuous at boundary and only crosses zero when inside window
		eps := 1e-6
		margin := absY - yWindow + eps
		if dx >= 0 {
			return margin
		}
		return -margin
	}
}

// EventCollision returns event for Moon surface collision.
func EventCollision(t float64, s State, phiS0 float64) float64 {
	return math.Hypot(s.Pos.X-x2, s.Pos.Y) - rMoon
}

// EventCapture returns event for lunar capture (negative energy).
func EventCapture(t float64, s State, phiS0 float64) float64 {
	rRel := math.Hypot(s.Pos.X-x2, s.Pos.Y)
	v2 := md3.Norm2(s.Vel)
	return 0.5*v2 - mu2/rRel
}

// DistanceToMoon returns distance from position to Moon center.
func DistanceToMoon(pos md3.Vec) float64 {
	return math.Hypot(pos.X-x2, pos.Y)
}

// Integrator performs RK45 (Dormand-Prince) integration for spacecraft trajectories.
type Integrator struct {
	T     float64
	State State
	PhiS0 float64
	Rates RatesFunc

	// Step control
	MinStep, MaxStep float64
	ATol, RTol       float64
}

// Dormand-Prince coefficients (RK45)
var (
	// Nodes
	dpC = [7]float64{0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1}

	// Matrix A (lower triangular)
	dpA = [7][6]float64{
		{},
		{1.0 / 5},
		{3.0 / 40, 9.0 / 40},
		{44.0 / 45, -56.0 / 15, 32.0 / 9},
		{19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
		{9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
		{35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84},
	}

	// 5th order weights
	dpB = [7]float64{35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0}

	// Error estimation weights (b - b*)
	dpE = [7]float64{
		35.0/384 - 5179.0/57600,
		0,
		500.0/1113 - 7571.0/16695,
		125.0/192 - 393.0/640,
		-2187.0/6784 + 92097.0/339200,
		11.0/84 - 187.0/2100,
		-1.0 / 40,
	}
)

// Step performs a single RK45 step with adaptive step size.
// Returns the new suggested step size.
func (ig *Integrator) Step(h float64) float64 {
	const (
		safety  = 0.9
		minFac  = 0.2
		maxFac  = 10.0
		order   = 5.0
	)

	t := ig.T
	s := ig.State
	phiS0 := ig.PhiS0
	rates := ig.Rates

	for {
		// Compute k values
		var kPos, kVel [7]md3.Vec
		var kM [7]float64

		kPos[0], kVel[0], kM[0] = rates(t, s, phiS0)

		for i := 1; i < 7; i++ {
			ti := t + dpC[i]*h
			var si State

			for j := 0; j < i; j++ {
				si.Pos = md3.Add(si.Pos, md3.Scale(h*dpA[i][j], kPos[j]))
				si.Vel = md3.Add(si.Vel, md3.Scale(h*dpA[i][j], kVel[j]))
				si.Mass += h * dpA[i][j] * kM[j]
			}
			si.Pos = md3.Add(s.Pos, si.Pos)
			si.Vel = md3.Add(s.Vel, si.Vel)
			si.Mass += s.Mass

			kPos[i], kVel[i], kM[i] = rates(ti, si, phiS0)
		}

		// Compute 5th order solution and error estimate
		var newPos, newVel, errPos, errVel md3.Vec
		var newM, errM float64

		for i := 0; i < 7; i++ {
			newPos = md3.Add(newPos, md3.Scale(h*dpB[i], kPos[i]))
			newVel = md3.Add(newVel, md3.Scale(h*dpB[i], kVel[i]))
			newM += h * dpB[i] * kM[i]

			errPos = md3.Add(errPos, md3.Scale(h*dpE[i], kPos[i]))
			errVel = md3.Add(errVel, md3.Scale(h*dpE[i], kVel[i]))
			errM += h * dpE[i] * kM[i]
		}

		newPos = md3.Add(s.Pos, newPos)
		newVel = md3.Add(s.Vel, newVel)
		newM += s.Mass

		// Error norm (scaled)
		scalePos := ig.ATol + ig.RTol*math.Max(md3.Norm(s.Pos), md3.Norm(newPos))
		scaleVel := ig.ATol + ig.RTol*math.Max(md3.Norm(s.Vel), md3.Norm(newVel))
		scaleM := ig.ATol + ig.RTol*math.Max(math.Abs(s.Mass), math.Abs(newM))

		errNorm := math.Sqrt((md3.Norm2(errPos)/(scalePos*scalePos) +
			md3.Norm2(errVel)/(scaleVel*scaleVel) +
			errM*errM/(scaleM*scaleM)) / 7)

		// Step accepted?
		if errNorm <= 1 {
			ig.T = t + h
			ig.State = State{Pos: newPos, Vel: newVel, Mass: newM}

			// Compute new step size
			if errNorm == 0 {
				return math.Min(h*maxFac, ig.MaxStep)
			}
			factor := safety * math.Pow(errNorm, -1.0/order)
			factor = math.Max(minFac, math.Min(maxFac, factor))
			return math.Max(ig.MinStep, math.Min(ig.MaxStep, h*factor))
		}

		// Step rejected, reduce step size
		factor := safety * math.Pow(errNorm, -1.0/order)
		factor = math.Max(minFac, factor)
		h = math.Max(ig.MinStep, h*factor)

		if h <= ig.MinStep {
			// Accept with minimum step
			ig.T = t + h
			ig.State = State{Pos: newPos, Vel: newVel, Mass: newM}
			return ig.MinStep
		}
	}
}

// IntegrateUntil integrates until tf or an event triggers.
// Returns the index of triggered event (-1 if none) and final time.
func (ig *Integrator) IntegrateUntil(tf float64, events []EventFunc) (eventIdx int, eventTime float64) {
	h := ig.MaxStep
	eventIdx = -1

	// Evaluate events at start
	prevEvents := make([]float64, len(events))
	for i, ev := range events {
		prevEvents[i] = ev(ig.T, ig.State, ig.PhiS0)
	}

	for ig.T < tf {
		// Limit step to not overshoot
		if ig.T+h > tf {
			h = tf - ig.T
		}

		tPrev := ig.T
		sPrev := ig.State
		h = ig.Step(h)

		// Check events for sign change
		for i, ev := range events {
			curr := ev(ig.T, ig.State, ig.PhiS0)
			if prevEvents[i]*curr < 0 {
				// Sign change detected, find root with bisection
				eventTime = bisectEvent(tPrev, ig.T, sPrev, ig.State, ig.PhiS0, ev)
				eventIdx = i

				// Interpolate state at event time (linear approximation)
				alpha := (eventTime - tPrev) / (ig.T - tPrev)
				ig.State = State{
					Pos:  md3.Add(md3.Scale(1-alpha, sPrev.Pos), md3.Scale(alpha, ig.State.Pos)),
					Vel:  md3.Add(md3.Scale(1-alpha, sPrev.Vel), md3.Scale(alpha, ig.State.Vel)),
					Mass: (1-alpha)*sPrev.Mass + alpha*ig.State.Mass,
				}
				ig.T = eventTime
				return eventIdx, eventTime
			}
			prevEvents[i] = curr
		}
	}

	return -1, ig.T
}

// bisectEvent finds the time of event crossing using bisection.
func bisectEvent(t0, t1 float64, s0, s1 State, phiS0 float64, ev EventFunc) float64 {
	const maxIter = 50
	const tol = 1e-10

	for i := 0; i < maxIter; i++ {
		tm := 0.5 * (t0 + t1)
		if t1-t0 < tol {
			return tm
		}

		// Linear interpolation of state
		alpha := (tm - t0) / (t1 - t0)
		sm := State{
			Pos:  md3.Add(md3.Scale(1-alpha, s0.Pos), md3.Scale(alpha, s1.Pos)),
			Vel:  md3.Add(md3.Scale(1-alpha, s0.Vel), md3.Scale(alpha, s1.Vel)),
			Mass: (1-alpha)*s0.Mass + alpha*s1.Mass,
		}

		vm := ev(tm, sm, phiS0)
		v0 := ev(t0, s0, phiS0)

		if v0*vm < 0 {
			t1 = tm
			s1 = sm
		} else {
			t0 = tm
			s0 = sm
		}
	}

	return 0.5 * (t0 + t1)
}

// Calculate runs the full three-phase trajectory simulation.
func (tr *Trajectory) Calculate() Result {
	startTime := time.Now()
	result := Result{}
	state := tr.InitialState()
	phiS0 := tr.PhiS0

	// Phase 1: Thrust until Jacobi threshold
	ig := Integrator{
		T:       0,
		State:   state,
		PhiS0:   phiS0,
		Rates:   RatesThrust(tr.Thruster),
		MinStep: 1,
		MaxStep: tr.MaxStep,
		ATol:    tr.Tol,
		RTol:    1e-9,
	}

	tf1 := float64(days * 360 * 4) // 4 years max
	evJacobi := EventJacobi(tr.JacobiThr)
	evIdx, evTime := ig.IntegrateUntil(tf1, []EventFunc{evJacobi})

	fmt.Printf("Fase 1 completada, t_eventos: [array([%.8f])]\n", evTime)
	if evIdx >= 0 {
		s := ig.State
		fmt.Printf("[Fase 1] Evento jacobiC_local disparado en t = [%.8f] con Jacobi\n", evTime)
		fmt.Printf("[Fase 1] Estado en evento: [[ %.8e %.8e %.8e %.8e\n   %.8e]]\n",
			s.Pos.X, s.Pos.Y, s.Vel.X, s.Vel.Y, s.Mass)
	} else {
		fmt.Println("[Fase 1] No se disparó el evento jacobiC_local")
		result.TotalTime = ig.T
		result.FinalMass = ig.State.Mass
		return result
	}
	t1 := ig.T // Cumulative time after phase 1

	// Phase 2: Coast until L1
	ig.PhiS0 = phiS0 - (nS-W)*t1
	ig.Rates = RatesCoast

	tf2 := ig.T + float64(days*650)
	evL1 := EventL1(55000)
	evIdx, evTime = ig.IntegrateUntil(tf2, []EventFunc{evL1})

	fmt.Printf("Fase 2 completada, tiempo de evento: [array([%.8f])]\n", evTime)
	fmt.Printf("[Fase 2] Estado: pos=(%.2f, %.2f), vel=(%.6f, %.6f), L1x=%.2f\n",
		ig.State.Pos.X, ig.State.Pos.Y, ig.State.Vel.X, ig.State.Vel.Y, L1x)
	if evIdx < 0 {
		fmt.Println("No se logró llegar a L1")
		result.TotalTime = ig.T
		result.FinalMass = ig.State.Mass
		result.ReachedL1 = false
		return result
	}
	fmt.Println("Se alcanzo L1 CORRECTAMENTE")
	result.ReachedL1 = true
	t2 := ig.T // Cumulative time after phase 2

	// Phase 3: Braking until capture or collision
	ig.PhiS0 = phiS0 - (nS-W)*t2
	ig.Rates = RatesBrake(tr.Thruster)

	tf3 := ig.T + float64(days*180)
	evJacobiC1 := EventJacobi(C1)
	evIdx, _ = ig.IntegrateUntil(tf3, []EventFunc{evJacobiC1, EventCollision, EventCapture})

	// Check capture condition (E < 0 throughout phase 3)
	rRel := DistanceToMoon(ig.State.Pos)
	v2 := md3.Norm2(ig.State.Vel)
	E := 0.5*v2 - mu2/rRel

	if E < 0 {
		fmt.Printf("La condición de captura (E < 0) se cumple - phi %.1f\n", tr.Phi*180/math.Pi)
		result.Captured = true
	} else {
		fmt.Printf("La condición de captura NO - %.1f\n", tr.Phi*180/math.Pi)
	}

	result.InSOI = rRel <= 55000

	switch evIdx {
	case 1: // Collision
		result.Collided = true
		fmt.Printf("Choco la LUNA phi - %.1f\n", tr.Phi*180/math.Pi)
	}

	result.TotalTime = ig.T
	result.FinalMass = ig.State.Mass

	execTime := time.Since(startTime).Seconds()
	fmt.Printf("{'phi': %.1f, 'jacobi_thr': %.3f, 'phiS0': %.0f, 'd0': %.0f, 'tiempo_total': %.8f, 'masa_final': %.8f, 'exito': %v, 'SOI': %v, 'time_exec': %.6f}\n",
		tr.Phi*180/math.Pi, tr.JacobiThr, tr.PhiS0*180/math.Pi, tr.D0,
		result.TotalTime/float64(days), result.FinalMass,
		result.Captured, result.InSOI, execTime)

	return result
}

func main() {
	// Print derived constants for comparison with Python
	fmt.Printf("Constants: W=%.10e, nS=%.10e, mu=%.2f, muS=%.10e\n", W, nS, mu, muS)
	fmt.Printf("Positions: x1=%.2f, x2=%.2f, L1x=%.2f\n", x1, x2, L1x)

	// Match Python: phi=295.5, phiS0=30, d0=37000, jacobi_thr=-1.639
	tr := Trajectory{
		D0:        37000,
		Phi:       295.5 * math.Pi / 180, // [rad]
		Gamma:     0,
		PhiS0:     30 * math.Pi / 180, // [rad]
		M0:        12,
		Thruster:  Thruster{Thrust: 4 * 0.00000045, Isp: 1650},
		JacobiThr: -1.639,
		Tol:       1e-12,
		MaxStep:   450,
	}

	// Print initial state for comparison
	s0 := tr.InitialState()
	fmt.Printf("Initial state: pos=(%.2f, %.2f), vel=(%.6f, %.6f), mass=%.2f\n",
		s0.Pos.X, s0.Pos.Y, s0.Vel.X, s0.Vel.Y, s0.Mass)

	tr.Calculate()
}
