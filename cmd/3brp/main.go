package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/soypat/geometry/md3"
)

// Physical constants for Earth-Moon system (3-body, no Sun).
const (
	days   = 24 * 3600 // [s/day]
	r12    = 384400.0  // [km] Earth-Moon distance
	m1     = 5974e21   // [kg] Earth mass
	m2     = 7348e19   // [kg] Moon mass
	mu1    = 398600.0  // [km³/s²] Earth gravitational parameter
	mu2    = 4903.02   // [km³/s²] Moon gravitational parameter
	rMoon  = 1737.0    // [km] Moon radius
	rEarth = 6378.0    // [km] Earth radius
	L1dist = 321710.0  // [km] L1 distance from Earth center
	C1     = -1.676    // Jacobi constant for braking phase

	enablePrinting = false
)

// Derived constants.
var (
	mu  = mu1 + mu2
	pi1 = m1 / (m1 + m2)
	pi2 = m2 / (m1 + m2)
	W   = math.Sqrt(mu / (r12 * r12 * r12))
	x1  = -pi2 * r12 // [km] Earth x-position in rotating frame
	x2  = pi1 * r12  // [km] Moon x-position in rotating frame
)

// State represents spacecraft state in the rotating Earth-Moon frame.
type State struct {
	Pos  md3.Vec // [km]
	Vel  md3.Vec // [km/s]
	Mass float64 // [kg]
}

// Thruster defines propulsion parameters.
type Thruster struct {
	Thrust float64 // [kN]
	Isp    float64 // [s]
}

// MassRate returns mass flow rate [kg/s] (negative).
func (th Thruster) MassRate() float64 {
	const g0 = 9.807e-3 // [km/s²]
	return -th.Thrust / (g0 * th.Isp)
}

// RatesFunc computes state derivatives for 3-body problem.
type RatesFunc func(t float64, s State) (dPos, dVel md3.Vec, dm float64)

// Record stores a single integration sample for output.
type Record struct {
	Phase int
	T     float64
	State
}

// Array returns the record as [phase, t, x, y, vx, vy, m].
func (r Record) Array() [7]float64 {
	return [7]float64{float64(r.Phase), r.T, r.Pos.X, r.Pos.Y, r.Vel.X, r.Vel.Y, r.Mass}
}

// EventFunc returns a value that crosses zero at an event.
type EventFunc func(t float64, s State) float64

// Accel computes gravitational acceleration from Earth and Moon.
func Accel(pos md3.Vec) md3.Vec {
	rToEarth := md3.Sub(pos, md3.Vec{X: x1})
	rToMoon := md3.Sub(pos, md3.Vec{X: x2})
	d1 := md3.Norm(rToEarth)
	d2 := md3.Norm(rToMoon)
	d1_3 := d1 * d1 * d1
	d2_3 := d2 * d2 * d2
	return md3.Add(md3.Scale(-mu1/d1_3, rToEarth), md3.Scale(-mu2/d2_3, rToMoon))
}

// RatesThrust returns derivatives with thrust along velocity direction.
func RatesThrust(th Thruster) RatesFunc {
	return func(t float64, s State) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel
		aGrav := Accel(s.Pos)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(th.Thrust/(s.Mass*v), s.Vel)
		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return
	}
}

// RatesCoast returns derivatives with no thrust.
func RatesCoast(t float64, s State) (dPos, dVel md3.Vec, dm float64) {
	dPos = s.Vel
	aGrav := Accel(s.Pos)
	aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
	aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
	dVel = md3.Add(md3.Add(aGrav, aCoriolis), aCentrifugal)
	return
}

// RatesBrake returns derivatives with thrust opposing velocity.
func RatesBrake(th Thruster) RatesFunc {
	return func(t float64, s State) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel
		aGrav := Accel(s.Pos)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(-th.Thrust/(s.Mass*v), s.Vel)
		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return
	}
}

// JacobiConstant computes the Jacobi integral for CR3BP.
func JacobiConstant(s State) float64 {
	v2 := md3.Norm2(s.Vel)
	d1 := math.Hypot(s.Pos.X-x1, s.Pos.Y)
	d2 := math.Hypot(s.Pos.X-x2, s.Pos.Y)
	return 0.5*v2 - 0.5*W*W*(s.Pos.X*s.Pos.X+s.Pos.Y*s.Pos.Y) - mu1/d1 - mu2/d2
}

// DistanceToEarth returns distance from pos to Earth center.
func DistanceToEarth(pos md3.Vec) float64 {
	return math.Hypot(pos.X-x1, pos.Y)
}

// DistanceToMoon returns distance from pos to Moon center.
func DistanceToMoon(pos md3.Vec) float64 {
	return math.Hypot(pos.X-x2, pos.Y)
}

// EventJacobi triggers when Jacobi constant crosses threshold.
func EventJacobi(threshold float64) EventFunc {
	return func(t float64, s State) float64 {
		return JacobiConstant(s) - threshold
	}
}

// EventL1 triggers when distance from Earth center reaches L1.
func EventL1(t float64, s State) float64 {
	dx := s.Pos.X - x1
	return dx*dx + s.Pos.Y*s.Pos.Y - L1dist*L1dist
}

// EventCollision triggers when spacecraft hits Moon surface.
func EventCollision(t float64, s State) float64 {
	dx := s.Pos.X - x2
	return dx*dx + s.Pos.Y*s.Pos.Y - rMoon*rMoon
}

// EventCapture triggers when specific orbital energy w.r.t. Moon goes negative.
func EventCapture(t float64, s State) float64 {
	rRel := DistanceToMoon(s.Pos)
	v2 := md3.Norm2(s.Vel)
	return 0.5*v2 - mu2/rRel
}

// Integrator performs RK45 (Dormand-Prince) integration for 3-body trajectories.
type Integrator struct {
	T     float64
	State State
	Rates RatesFunc

	MinStep, MaxStep float64
	ATol, RTol       float64
	Phase            int
	Hist             []Record
}

func (ig *Integrator) record() {
	ig.Hist = append(ig.Hist, Record{Phase: ig.Phase, T: ig.T, State: ig.State})
}

// Dormand-Prince coefficients.
var (
	dpC = [7]float64{0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1}
	dpA = [7][6]float64{
		{},
		{1.0 / 5},
		{3.0 / 40, 9.0 / 40},
		{44.0 / 45, -56.0 / 15, 32.0 / 9},
		{19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
		{9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
		{35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84},
	}
	dpB = [7]float64{35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0}
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

// Step performs a single adaptive RK45 step. Returns next suggested step size.
func (ig *Integrator) Step(h float64) float64 {
	const (
		safety = 0.9
		minFac = 0.2
		maxFac = 10.0
		order  = 5.0
	)

	t := ig.T
	s := ig.State
	rates := ig.Rates

	for {
		var kPos, kVel [7]md3.Vec
		var kM [7]float64

		kPos[0], kVel[0], kM[0] = rates(t, s)

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
			kPos[i], kVel[i], kM[i] = rates(ti, si)
		}

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

		scalePos := ig.ATol + ig.RTol*math.Max(md3.Norm(s.Pos), md3.Norm(newPos))
		scaleVel := ig.ATol + ig.RTol*math.Max(md3.Norm(s.Vel), md3.Norm(newVel))
		scaleM := ig.ATol + ig.RTol*math.Max(math.Abs(s.Mass), math.Abs(newM))
		errNorm := math.Sqrt((md3.Norm2(errPos)/(scalePos*scalePos) +
			md3.Norm2(errVel)/(scaleVel*scaleVel) +
			errM*errM/(scaleM*scaleM)) / 7)

		if errNorm <= 1 {
			ig.T = t + h
			ig.State = State{Pos: newPos, Vel: newVel, Mass: newM}
			if errNorm == 0 {
				return math.Min(h*maxFac, ig.MaxStep)
			}
			factor := safety * math.Pow(errNorm, -1.0/order)
			factor = math.Max(minFac, math.Min(maxFac, factor))
			return math.Max(ig.MinStep, math.Min(ig.MaxStep, h*factor))
		}

		factor := safety * math.Pow(errNorm, -1.0/order)
		factor = math.Max(minFac, factor)
		h = math.Max(ig.MinStep, h*factor)
		if h <= ig.MinStep {
			ig.T = t + h
			ig.State = State{Pos: newPos, Vel: newVel, Mass: newM}
			return ig.MinStep
		}
	}
}

// IntegrateUntil integrates until tf or an event triggers.
// Returns triggered event index (-1 if none).
func (ig *Integrator) IntegrateUntil(tf float64, events []EventFunc) (eventIdx int) {
	ig.record()
	h := ig.MaxStep
	prevEv := make([]float64, len(events))
	for i, ev := range events {
		prevEv[i] = ev(ig.T, ig.State)
	}
	for ig.T < tf {
		if ig.T+h > tf {
			h = tf - ig.T
		}
		tPrev := ig.T
		sPrev := ig.State
		h = ig.Step(h)

		for i, ev := range events {
			curr := ev(ig.T, ig.State)
			if prevEv[i]*curr < 0 {
				evTime := bisectEvent(tPrev, ig.T, sPrev, ig.State, ev)
				alpha := (evTime - tPrev) / (ig.T - tPrev)
				ig.State = State{
					Pos:  md3.Add(md3.Scale(1-alpha, sPrev.Pos), md3.Scale(alpha, ig.State.Pos)),
					Vel:  md3.Add(md3.Scale(1-alpha, sPrev.Vel), md3.Scale(alpha, ig.State.Vel)),
					Mass: (1-alpha)*sPrev.Mass + alpha*ig.State.Mass,
				}
				ig.T = evTime
				ig.record()
				return i
			}
			prevEv[i] = curr
		}
		ig.record()
	}
	return -1
}

func bisectEvent(t0, t1 float64, s0, s1 State, ev EventFunc) float64 {
	const maxIter = 50
	const tol = 1e-10
	for range maxIter {
		tm := 0.5 * (t0 + t1)
		if t1-t0 < tol {
			return tm
		}
		alpha := (tm - t0) / (t1 - t0)
		sm := State{
			Pos:  md3.Add(md3.Scale(1-alpha, s0.Pos), md3.Scale(alpha, s1.Pos)),
			Vel:  md3.Add(md3.Scale(1-alpha, s0.Vel), md3.Scale(alpha, s1.Vel)),
			Mass: (1-alpha)*s0.Mass + alpha*s1.Mass,
		}
		if ev(t0, s0)*ev(tm, sm) < 0 {
			t1 = tm
			s1 = sm
		} else {
			t0 = tm
			s0 = sm
		}
	}
	return 0.5 * (t0 + t1)
}

func writeCSV(path string, records []Record) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, r := range records {
		a := r.Array()
		fmt.Fprintf(w, "%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n",
			int(a[0]), a[1], a[2], a[3], a[4], a[5], a[6])
	}
	if err := w.Flush(); err != nil {
		panic(err)
	}
}

// Trajectory holds initial conditions for a 3-body transfer.
type Trajectory struct {
	D0    float64 // [km] Altitude above Earth
	Phi   float64 // [rad] Initial angle
	Gamma float64 // [rad] Flight path angle
	M0    float64 // [kg] Initial mass

	Thruster  Thruster
	JacobiThr float64 // Jacobi threshold for phase 1
	Tol       float64 // Integration tolerance
}

// InitialState computes state from trajectory parameters.
func (tr *Trajectory) InitialState() State {
	r0 := rEarth + tr.D0
	v0 := math.Sqrt(mu1/r0) - W*r0
	sinPhi, cosPhi := math.Sincos(tr.Phi)
	sinGam, cosGam := math.Sincos(tr.Gamma)
	return State{
		Pos: md3.Vec{
			X: r0*cosPhi + x1,
			Y: r0 * sinPhi,
		},
		Vel: md3.Vec{
			X: v0 * (sinGam*cosPhi - cosGam*sinPhi),
			Y: v0 * (sinGam*sinPhi + cosGam*cosPhi),
		},
		Mass: tr.M0,
	}
}

// Calculate runs the 4-phase trajectory simulation.
func (tr *Trajectory) Calculate() {
	startTime := time.Now()
	s0 := tr.InitialState()

	ig := Integrator{
		T:       0,
		State:   s0,
		Rates:   RatesThrust(tr.Thruster),
		MinStep: 1,
		MaxStep: 450,
		ATol:    tr.Tol,
		RTol:    1e-9,
	}

	// Phase 1: Thrust until Jacobi threshold.
	ig.Phase = 1
	tf1 := float64(days * 360 * 4)
	evIdx := ig.IntegrateUntil(tf1, []EventFunc{EventJacobi(tr.JacobiThr)})
	printf("Fase 1 completada, tiempo de evento: %.2f, shape: %d\n", ig.T, evIdx)
	if evIdx < 0 {
		printf("Fase 1: no se alcanzó umbral Jacobi\n")
		return
	}
	printState("Fase 1", ig.T, ig.State)

	// Phase 2: Coast until L1 distance from Earth.
	ig.Phase = 2
	ig.Rates = RatesCoast
	ig.MaxStep = 200
	tf2 := ig.T + float64(days*650)
	evIdx = ig.IntegrateUntil(tf2, []EventFunc{EventL1})
	printf("Fase 2 completada, tiempo de evento: %.2f\n", ig.T)
	if evIdx < 0 {
		printf("No se logró llegar a L1\n")
		return
	}
	printf("Se alcanzó L1 CORRECTAMENTE\n")
	printState("Fase 2", ig.T, ig.State)

	// Phase 3: Brake until Jacobi reaches C1.
	ig.Phase = 3
	ig.Rates = RatesBrake(tr.Thruster)
	ig.MaxStep = 100
	tf3 := ig.T + float64(days*2000)
	evIdx = ig.IntegrateUntil(tf3, []EventFunc{EventJacobi(C1)})
	printf("Fase 3 completada, tiempo de evento: %.2f\n", ig.T)
	printState("Fase 3", ig.T, ig.State)

	// Phase 4: Coast post-brake (20 days).
	ig.Phase = 4
	ig.Rates = RatesCoast
	ig.MaxStep = 100
	tf4 := ig.T + float64(days*20)
	ig.IntegrateUntil(tf4, nil)
	printf("Fase 4 completada, tiempo final: %.2f\n", ig.T)
	printState("Fase 4", ig.T, ig.State)
	elapsed := time.Since(startTime).Seconds()
	fmt.Printf("Tiempo que demoró -- %.6f\n", elapsed)

	writeCSV("trajectory.csv", ig.Hist)

	// Capture energy check.
	rRel := DistanceToMoon(ig.State.Pos)
	v2 := md3.Norm2(ig.State.Vel)
	E := 0.5*v2 - mu2/rRel
	printf("Energía captura final: %.6e\n", E)

}

func printf(format string, a ...any) {
	if enablePrinting {
		fmt.Printf(format, a...)
	}
}

func printState(phase string, t float64, s State) {
	printf("[%s] t=%.2fs (%.2f days)  pos=(%.2f, %.2f)  vel=(%.6f, %.6f)  mass=%.4f\n",
		phase, t, t/float64(days), s.Pos.X, s.Pos.Y, s.Vel.X, s.Vel.Y, s.Mass)
}

func main() {
	// Match Python: phi=292.95343988881166, d0=37000, F=0.00000110, n=1, Isp=2150
	phi := 292.95343988881166
	tr := Trajectory{
		D0:        37000,
		Phi:       phi * math.Pi / 180,
		Gamma:     0,
		M0:        12,
		Thruster:  Thruster{Thrust: 1 * 0.00000110, Isp: 2150},
		JacobiThr: -1.63907788,
		Tol:       1e-12,
	}
	printf("phi=%.8f  F=%.10f  Isp=%d\n", phi, tr.Thruster.Thrust, 2150)
	s0 := tr.InitialState()
	printf("Initial: pos=(%.2f, %.2f) vel=(%.6f, %.6f) mass=%.2f\n",
		s0.Pos.X, s0.Pos.Y, s0.Vel.X, s0.Vel.Y, s0.Mass)
	tr.Calculate()
}
