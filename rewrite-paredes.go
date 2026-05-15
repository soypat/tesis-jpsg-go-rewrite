package main

import (
	"fmt"
	"math"
	"time"

	"github.com/soypat/geometry/md3"
)

// ParedesRecord stores one accepted RK45 step for trajectory output.
type ParedesRecord struct {
	Phase   int     // 1–4
	T       float64 // [s] time since mission start
	Pos     md3.Vec // [km] position in rotating frame
	Vel     md3.Vec // [km/s] velocity in rotating frame
	Mass    float64 // [kg]
	ErrNorm float64 // error norm from Step() — drives h_next
}

// GTOConfig holds initial conditions and simulation parameters for the
// GTO-to-lunar-capture trajectory (Earth+Moon CR3BP, no Sun).
type GTOConfig struct {
	HApogee   float64  // [km] apogee altitude above Earth surface
	HPerigee  float64  // [km] perigee altitude above Earth surface
	Phi       float64  // [rad] initial angle of apogee from rotating-frame x-axis
	Gamma     float64  // [rad] flight path angle (0 = tangential departure)
	M0        float64  // [kg] initial spacecraft mass
	Thruster  Thruster // reused from rewrite.go
	JacobiThr float64  // Jacobi constant threshold to end Phase 1
	Tol       float64  // ATol; RTol fixed at 1e-9
}

// GTOResult holds the output of the 4-phase GTO simulation.
type GTOResult struct {
	TotalTime float64 // [s] integrated mission time at end of Phase 4
	FinalMass float64 // [kg] spacecraft mass at end of Phase 4
	Success   bool    // true if all 4 phases completed
	ReachedL1 bool    // true if Phase 2 triggered the L1 event
	Phase1    []ParedesRecord
	Phase2    []ParedesRecord
	Phase3    []ParedesRecord
	Phase4    []ParedesRecord
}

// AccelEM computes gravitational acceleration from Earth and Moon only (no Sun).
// Distances use sqrt(X*X + Y*Y) to match Python's np.linalg.norm([X, Y]).
// The Y-component uses (mu1/r1³ + mu2/r2³)*y to match Python's factor-then-scale.
func AccelEM(pos md3.Vec) md3.Vec {
	dx1 := pos.X - x1
	dx2 := pos.X - x2
	d1 := math.Sqrt(dx1*dx1 + pos.Y*pos.Y) // matches: np.linalg.norm([x+pi2*r12, y])
	d2 := math.Sqrt(dx2*dx2 + pos.Y*pos.Y) // matches: np.linalg.norm([x-pi1*r12, y])
	d1_3 := d1 * d1 * d1
	d2_3 := d2 * d2 * d2
	// X: each term is separate — matches Python's: - mu1*(x-x1)/r1³ - mu2*(x-x2)/r2³
	// Y: combined rate*(y) — matches Python's: - (mu1/r1³ + mu2/r2³)*y
	gY := -(mu1/d1_3 + mu2/d2_3) * pos.Y
	return md3.Vec{
		X: -mu1/d1_3*dx1 - mu2/d2_3*dx2,
		Y: gY,
	}
}

// JacobiConstantEM computes the Jacobi constant for the Earth+Moon CR3BP (no Sun).
// J = 0.5v² - 0.5W²(x²+y²) - mu1/r1 - mu2/r2
func JacobiConstantEM(s State) float64 {
	v2 := md3.Norm2(s.Vel)
	pos := s.Pos
	d1 := math.Hypot(pos.X-x1, pos.Y)
	d2 := math.Hypot(pos.X-x2, pos.Y)
	return 0.5*v2 - 0.5*W*W*(pos.X*pos.X+pos.Y*pos.Y) - mu1/d1 - mu2/d2
}

// paredesGravity computes distances and per-body gravity coefficients used by
// all three rates functions. Matches Python's np.linalg.norm and r**3 style.
func paredesGravity(x, y float64) (r1_3, r2_3 float64) {
	dx1 := x - x1
	dx2 := x - x2
	r1 := math.Sqrt(dx1*dx1 + y*y) // np.linalg.norm([x+pi2*r12, y])
	r2 := math.Sqrt(dx2*dx2 + y*y) // np.linalg.norm([x-pi1*r12, y])
	return r1 * r1 * r1, r2 * r2 * r2
}

// RatesThrustEM returns a RatesFunc for prograde thrust in Earth+Moon CR3BP.
// Mirrors Python rates(t,f) sequential left-to-right evaluation exactly:
//
//	ax = 2*W*vy + W²*x - mu1*(x-x1)/r1³ - mu2*(x-x2)/r2³ + (T/m)*(vx/v)
//	ay = -2*W*vx + W²*y - (mu1/r1³ + mu2/r2³)*y + (T/m)*(vy/v)
func RatesThrustEM(th Thruster) RatesFunc {
	return func(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
		x, y := s.Pos.X, s.Pos.Y
		vx, vy, m := s.Vel.X, s.Vel.Y, s.Mass
		r1_3, r2_3 := paredesGravity(x, y)
		v := math.Sqrt(vx*vx + vy*vy) // np.linalg.norm([vx, vy])
		tmv := th.Thrust / (m * v)
		// Python sequential left-to-right for each axis:
		ax := ((2*W*vy + W*W*x) - mu1*(x-x1)/r1_3) - mu2*(x-x2)/r2_3 + tmv*vx
		ay := (-2*W*vx + W*W*y) - (mu1/r1_3+mu2/r2_3)*y + tmv*vy
		dPos = s.Vel
		dVel = md3.Vec{X: ax, Y: ay}
		dm = th.MassRate()
		return
	}
}

// RatesCoastEM is a RatesFunc for unpowered coasting in Earth+Moon CR3BP.
// Mirrors Python rates0(t,f) sequential left-to-right evaluation exactly.
func RatesCoastEM(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
	x, y := s.Pos.X, s.Pos.Y
	vx, vy := s.Vel.X, s.Vel.Y
	r1_3, r2_3 := paredesGravity(x, y)
	ax := ((2*W*vy + W*W*x) - mu1*(x-x1)/r1_3) - mu2*(x-x2)/r2_3
	ay := (-2*W*vx + W*W*y) - (mu1/r1_3+mu2/r2_3)*y
	dPos = s.Vel
	dVel = md3.Vec{X: ax, Y: ay}
	return
}

// RatesBrakeEM returns a RatesFunc for retrograde braking in Earth+Moon CR3BP.
// Mirrors Python rates_1(t,f) sequential left-to-right evaluation exactly.
func RatesBrakeEM(th Thruster) RatesFunc {
	return func(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
		x, y := s.Pos.X, s.Pos.Y
		vx, vy, m := s.Vel.X, s.Vel.Y, s.Mass
		r1_3, r2_3 := paredesGravity(x, y)
		v := math.Sqrt(vx*vx + vy*vy)
		tmv := -th.Thrust / (m * v) // negative: retrograde
		ax := ((2*W*vy + W*W*x) - mu1*(x-x1)/r1_3) - mu2*(x-x2)/r2_3 + tmv*vx
		ay := (-2*W*vx + W*W*y) - (mu1/r1_3+mu2/r2_3)*y + tmv*vy
		dPos = s.Vel
		dVel = md3.Vec{X: ax, Y: ay}
		dm = th.MassRate()
		return
	}
}

// EventJacobiEM returns an EventFunc that triggers when the EM-only Jacobi
// constant crosses threshold. Matches Python jacobiC and jacobiC1 behavior.
func EventJacobiEM(threshold float64) EventFunc {
	return func(t float64, s State, _ float64) float64 {
		return JacobiConstantEM(s) - threshold
	}
}

// EventL1EM triggers when the spacecraft's distance from Earth center equals
// L1dist (321710 km). Matches Python lagrian1(t, y).
func EventL1EM(t float64, s State, _ float64) float64 {
	return math.Hypot(s.Pos.X-x1, s.Pos.Y) - L1dist
}

// GTOInitialState computes spacecraft state at GTO apogee in the rotating frame.
// Uses vis-viva at apogee: v = sqrt(mu1*(1-e)/rApogee) minus frame rotation W*rApogee.
func (cfg *GTOConfig) GTOInitialState() State {
	rApogee := rEarth + cfg.HApogee
	rPerigee := rEarth + cfg.HPerigee
	e := (rApogee - rPerigee) / (rApogee + rPerigee)
	v0 := math.Sqrt(mu1*(1-e)/rApogee) - W*rApogee

	sinPhi, cosPhi := math.Sincos(cfg.Phi)
	sinGam, cosGam := math.Sincos(cfg.Gamma)

	return State{
		Pos:  md3.Vec{X: rApogee*cosPhi + x1, Y: rApogee * sinPhi},
		Vel:  md3.Vec{X: v0 * (sinGam*cosPhi - cosGam*sinPhi), Y: v0 * (sinGam*sinPhi + cosGam*cosPhi)},
		Mass: cfg.M0,
	}
}

// SelectInitialStep computes a safe first step size following Hairer, Norsett
// & Wanner §II.4 — the same algorithm used by scipy's select_initial_step.
// Uses per-component scaling (sc_i = ATol + |y_i|·RTol) unlike Step() which
// uses vector-norm scaling; this matches scipy's step-size controller exactly.
func (ig *Integrator) SelectInitialStep() float64 {
	const (
		errOrder = 4.0 // RK45 error-estimator order (scipy: error_estimator_order)
		nComp    = 5.0 // active state components for this 2D problem: x,y,vx,vy,m
		// (z and vz are always 0; including them would dilute the rms by sqrt(5/7)
		// and shift the initial h ~3.4% away from scipy's value)
	)
	s := ig.State
	dPos0, dVel0, dm0 := ig.Rates(ig.T, s, ig.PhiS0)

	// Per-component scale factors.
	atol, rtol := ig.ATol, ig.RTol
	sc := [7]float64{
		atol + math.Abs(s.Pos.X)*rtol, atol + math.Abs(s.Pos.Y)*rtol, atol + math.Abs(s.Pos.Z)*rtol,
		atol + math.Abs(s.Vel.X)*rtol, atol + math.Abs(s.Vel.Y)*rtol, atol + math.Abs(s.Vel.Z)*rtol,
		atol + math.Abs(s.Mass)*rtol,
	}
	rmsNorm := func(px, py, pz, vx, vy, vz, m float64) float64 {
		return math.Sqrt((px*px+py*py+pz*pz+vx*vx+vy*vy+vz*vz+m*m)/nComp)
	}

	d0 := rmsNorm(s.Pos.X/sc[0], s.Pos.Y/sc[1], s.Pos.Z/sc[2],
		s.Vel.X/sc[3], s.Vel.Y/sc[4], s.Vel.Z/sc[5], s.Mass/sc[6])
	d1 := rmsNorm(dPos0.X/sc[0], dPos0.Y/sc[1], dPos0.Z/sc[2],
		dVel0.X/sc[3], dVel0.Y/sc[4], dVel0.Z/sc[5], dm0/sc[6])

	var h0 float64
	if d0 < 1e-5 || d1 < 1e-5 {
		h0 = 1e-6
	} else {
		h0 = 0.01 * d0 / d1
	}

	// One explicit Euler probe step to estimate second derivative.
	s1 := State{
		Pos:  md3.Add(s.Pos, md3.Scale(h0, dPos0)),
		Vel:  md3.Add(s.Vel, md3.Scale(h0, dVel0)),
		Mass: s.Mass + h0*dm0,
	}
	dPos1, dVel1, dm1 := ig.Rates(ig.T+h0, s1, ig.PhiS0)
	ddPos, ddVel, ddm := md3.Sub(dPos1, dPos0), md3.Sub(dVel1, dVel0), dm1-dm0
	d2 := rmsNorm(ddPos.X/sc[0], ddPos.Y/sc[1], ddPos.Z/sc[2],
		ddVel.X/sc[3], ddVel.Y/sc[4], ddVel.Z/sc[5], ddm/sc[6]) / h0

	var h1 float64
	if maxD := math.Max(d1, d2); maxD <= 1e-5 {
		h1 = math.Max(1e-6, h0*1e-3)
	} else {
		h1 = math.Pow(0.01/maxD, 1/(errOrder+1))
	}
	return math.Min(100*h0, h1)
}

// IntegrateUntilRecording integrates until tf or an event triggers, recording
// every accepted RK45 step. Returns the triggered event index (-1 if none)
// and all records for this phase including the refined event endpoint.
func (ig *Integrator) IntegrateUntilRecording(tf float64, events []EventFunc, phaseNum int) (eventIdx int, records []ParedesRecord) {
	h := math.Min(ig.SelectInitialStep(), ig.MaxStep)
	eventIdx = -1

	records = append(records, ParedesRecord{Phase: phaseNum, T: ig.T, Pos: ig.State.Pos, Vel: ig.State.Vel, Mass: ig.State.Mass})

	prevEvents := make([]float64, len(events))
	for i, ev := range events {
		prevEvents[i] = ev(ig.T, ig.State, ig.PhiS0)
	}

	for ig.T < tf {
		if ig.T+h > tf {
			h = tf - ig.T
		}

		tPrev := ig.T
		sPrev := ig.State
		h = ig.Step(h)

		records = append(records, ParedesRecord{Phase: phaseNum, T: ig.T, Pos: ig.State.Pos, Vel: ig.State.Vel, Mass: ig.State.Mass, ErrNorm: ig.LastErrNorm})

		for i, ev := range events {
			curr := ev(ig.T, ig.State, ig.PhiS0)
			if prevEvents[i]*curr < 0 {
				eventTime := bisectEvent(tPrev, ig.T, sPrev, ig.State, ig.PhiS0, ev)
				eventIdx = i

				alpha := (eventTime - tPrev) / (ig.T - tPrev)
				ig.State = State{
					Pos:  md3.Add(md3.Scale(1-alpha, sPrev.Pos), md3.Scale(alpha, ig.State.Pos)),
					Vel:  md3.Add(md3.Scale(1-alpha, sPrev.Vel), md3.Scale(alpha, ig.State.Vel)),
					Mass: (1-alpha)*sPrev.Mass + alpha*ig.State.Mass,
				}
				ig.T = eventTime

				records[len(records)-1] = ParedesRecord{Phase: phaseNum, T: ig.T, Pos: ig.State.Pos, Vel: ig.State.Vel, Mass: ig.State.Mass}
				return eventIdx, records
			}
			prevEvents[i] = curr
		}
	}

	return -1, records
}

// Calculate runs the 4-phase GTO-to-lunar-capture simulation.
func (cfg *GTOConfig) Calculate() GTOResult {
	result := GTOResult{}
	th := cfg.Thruster

	ig := Integrator{
		T:       0,
		State:   cfg.GTOInitialState(),
		PhiS0:   0,
		Rates:   RatesThrustEM(th),
		MinStep: 0,
		MaxStep: 450,
		ATol:    cfg.Tol,
		RTol:    1e-9,
	}
	s0 := ig.State
	debugf("[P0] IC: x=%.10e y=%.10e vx=%.10e vy=%.10e m=%.10f\n",
		s0.Pos.X, s0.Pos.Y, s0.Vel.X, s0.Vel.Y, s0.Mass)

	// Phase 1: prograde thrust until Jacobi threshold.
	evIdx, p1 := ig.IntegrateUntilRecording(float64(days*360*4), []EventFunc{EventJacobiEM(cfg.JacobiThr)}, 1)
	result.Phase1 = p1
	fmt.Printf("Fase 1 completada, tiempo de evento: [array([%.7f])] (5, %d)\n", ig.T, len(p1))
	debugf("[P1 end] t=%.10f x=%.10e y=%.10e vx=%.10e vy=%.10e m=%.10f\n",
		ig.T, ig.State.Pos.X, ig.State.Pos.Y, ig.State.Vel.X, ig.State.Vel.Y, ig.State.Mass)
	debugf("[P1 end] Jacobi=%.12f (thr=%.8f, residual=%.3e)\n",
		JacobiConstantEM(ig.State), cfg.JacobiThr, JacobiConstantEM(ig.State)-cfg.JacobiThr)
	debugf("[P1] steps=%d avg_step=%.1fs\n", len(p1)-1, ig.T/float64(len(p1)-1))
	for i := 1; i <= 5 && i < len(p1); i++ {
		debugf("[P1] step[%d] h=%.6f\n", i, p1[i].T-p1[i-1].T)
	}
	nextP1 := 0.0
	for _, r := range p1 {
		if r.T >= nextP1 {
			debugf("[P1 t=%7.2fd] x=%.8e y=%.8e vx=%.8e vy=%.8e\n",
				r.T/days, r.Pos.X, r.Pos.Y, r.Vel.X, r.Vel.Y)
			nextP1 += 20 * days
		}
	}

	// Bit-level step comparison: log-spaced indices for binary-search of first divergence.
	p1stepIdxs := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
		20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, len(p1) - 1}
	seen := map[int]bool{}
	for _, idx := range p1stepIdxs {
		if idx >= 0 && idx < len(p1) && !seen[idx] {
			seen[idx] = true
			r := p1[idx]
			var h float64
			if idx > 0 {
				h = r.T - p1[idx-1].T
			}
			debugf("[P1 s%5d] t=%.17e h=%.17e x=%.17e y=%.17e vx=%.17e vy=%.17e m=%.17e en=%.17e\n",
				idx, r.T, h, r.Pos.X, r.Pos.Y, r.Vel.X, r.Vel.Y, r.Mass, r.ErrNorm)
		}
	}

	if evIdx < 0 {
		result.TotalTime = ig.T
		result.FinalMass = ig.State.Mass
		return result
	}

	// Phase 2: coast until distance from Earth = L1dist.
	ig.Rates = RatesCoastEM
	ig.MaxStep = 200
	evIdx, p2 := ig.IntegrateUntilRecording(ig.T+float64(days*650), []EventFunc{EventL1EM}, 2)
	result.Phase2 = p2
	fmt.Printf("Fase 2 completada, tiempo de evento: [array([%.8f])] (5, %d)\n", ig.T, len(p2))
	// Print Phase 2 trajectory at 5-day intervals (post-hoc over records).
	nextPrint := p2[0].T
	for _, r := range p2 {
		if r.T >= nextPrint {
			debugf("[P2 t=%6.2fd] x=%.6e y=%.6e dist_earth=%.3f\n",
				r.T/days, r.Pos.X, r.Pos.Y,
				math.Hypot(r.Pos.X-x1, r.Pos.Y))
			nextPrint += 5 * days
		}
	}
	debugf("[P2 end] t=%.10f dist_earth=%.10f L1dist=%.1f residual=%.6e\n",
		ig.T, math.Hypot(ig.State.Pos.X-x1, ig.State.Pos.Y),
		L1dist, math.Hypot(ig.State.Pos.X-x1, ig.State.Pos.Y)-L1dist)
	if evIdx < 0 {
		fmt.Println("No se logró llegar a L1")
		result.TotalTime = ig.T
		result.FinalMass = ig.State.Mass
		return result
	}
	fmt.Println("Se alcanzó L1 CORRECTAMENTE")
	result.ReachedL1 = true

	// Phase 3: retrograde braking until Jacobi constant reaches C1.
	ig.Rates = RatesBrakeEM(th)
	ig.MaxStep = days // no max_step in Python — allow large adaptive steps
	evIdx, p3 := ig.IntegrateUntilRecording(ig.T+float64(days*25), []EventFunc{EventJacobiEM(C1)}, 3)
	result.Phase3 = p3
	fmt.Printf("Fase 3 completada, tiempo de evento: [array([%.8f])] (5, %d)\n", ig.T, len(p3))
	_ = evIdx

	// Phase 4: 20-day coast, no events.
	ig.Rates = RatesCoastEM
	ig.MaxStep = 100
	_, p4 := ig.IntegrateUntilRecording(ig.T+float64(days*20), nil, 4)
	result.Phase4 = p4
	fmt.Printf("Fase 4 completada, tiempo final: %.8f (5, %d)\n", ig.T, len(p4))

	result.TotalTime = ig.T
	result.FinalMass = ig.State.Mass
	result.Success = true
	return result
}

// RunParedes runs the Paredes GTO simulation with the default parameters
// from metodoparedesGTO.py and prints results matching Python output format.
func RunParedes() {
	start := time.Now()
	cfg := GTOConfig{
		HApogee:   37000,
		HPerigee:  1200,
		Phi:       0.7505211952744961 * math.Pi / 180,
		Gamma:     0,
		M0:        12,
		Thruster:  Thruster{Thrust: 4 * 0.000000450, Isp: 1650},
		JacobiThr: -1.63907788,
		Tol:       1e-12,
	}
	result := cfg.Calculate()
	fmt.Printf("Tiempo que demoró --  %v\n", time.Since(start).Seconds())
	fmt.Printf("Masa final  %.15f\n", result.FinalMass)
}
