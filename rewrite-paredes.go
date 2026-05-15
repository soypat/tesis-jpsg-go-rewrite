package main

import (
	"fmt"
	"math"
	"time"

	"github.com/soypat/geometry/md3"
)

// ParedesRecord stores one accepted RK45 step for trajectory output.
type ParedesRecord struct {
	Phase int     // 1–4
	T     float64 // [s] time since mission start
	Pos   md3.Vec // [km] position in rotating frame
	Vel   md3.Vec // [km/s] velocity in rotating frame
	Mass  float64 // [kg]
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
// Equivalent to the gravity terms in the Python rates/rates0/rates_1 functions.
func AccelEM(pos md3.Vec) md3.Vec {
	rToEarth := md3.Sub(pos, md3.Vec{X: x1})
	rToMoon := md3.Sub(pos, md3.Vec{X: x2})
	d1 := md3.Norm(rToEarth)
	d2 := md3.Norm(rToMoon)
	d1_3 := d1 * d1 * d1
	d2_3 := d2 * d2 * d2
	return md3.Add(
		md3.Scale(-mu1/d1_3, rToEarth),
		md3.Scale(-mu2/d2_3, rToMoon),
	)
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

// RatesThrustEM returns a RatesFunc for prograde thrust in Earth+Moon CR3BP.
// Matches Python rates(t, f): thrust aligned along velocity.
func RatesThrustEM(th Thruster) RatesFunc {
	return func(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel
		aGrav := AccelEM(s.Pos)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(th.Thrust/(s.Mass*v), s.Vel)
		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return dPos, dVel, dm
	}
}

// RatesCoastEM is a RatesFunc for unpowered coasting in Earth+Moon CR3BP.
// Matches Python rates0(t, f). phiS0 is unused (no Sun).
func RatesCoastEM(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
	dPos = s.Vel
	aGrav := AccelEM(s.Pos)
	aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
	aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
	dVel = md3.Add(md3.Add(aGrav, aCoriolis), aCentrifugal)
	return dPos, dVel, 0
}

// RatesBrakeEM returns a RatesFunc for retrograde braking in Earth+Moon CR3BP.
// Matches Python rates_1(t, f): thrust opposes velocity.
func RatesBrakeEM(th Thruster) RatesFunc {
	return func(t float64, s State, _ float64) (dPos, dVel md3.Vec, dm float64) {
		dPos = s.Vel
		aGrav := AccelEM(s.Pos)
		aCoriolis := md3.Vec{X: 2 * W * s.Vel.Y, Y: -2 * W * s.Vel.X}
		aCentrifugal := md3.Vec{X: W * W * s.Pos.X, Y: W * W * s.Pos.Y}
		v := md3.Norm(s.Vel)
		aThrust := md3.Scale(-th.Thrust/(s.Mass*v), s.Vel)
		dVel = md3.Add(md3.Add(aGrav, aCoriolis), md3.Add(aCentrifugal, aThrust))
		dm = th.MassRate()
		return dPos, dVel, dm
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

// IntegrateUntilRecording integrates until tf or an event triggers, recording
// every accepted RK45 step. Returns the triggered event index (-1 if none)
// and all records for this phase including the refined event endpoint.
func (ig *Integrator) IntegrateUntilRecording(tf float64, events []EventFunc, phaseNum int) (eventIdx int, records []ParedesRecord) {
	h := ig.MaxStep
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

		records = append(records, ParedesRecord{Phase: phaseNum, T: ig.T, Pos: ig.State.Pos, Vel: ig.State.Vel, Mass: ig.State.Mass})

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
		MinStep: 1,
		MaxStep: 450,
		ATol:    cfg.Tol,
		RTol:    1e-9,
	}

	// Phase 1: prograde thrust until Jacobi threshold.
	evIdx, p1 := ig.IntegrateUntilRecording(float64(days*360*4), []EventFunc{EventJacobiEM(cfg.JacobiThr)}, 1)
	result.Phase1 = p1
	fmt.Printf("Fase 1 completada, tiempo de evento: [array([%.7f])] (5, %d)\n", ig.T, len(p1))
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
